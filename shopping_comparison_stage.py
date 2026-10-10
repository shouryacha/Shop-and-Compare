"""
Shopping comparison stage: for one product, gather multiple Google Shopping
listings and have Nemotron classify EACH ONE independently against the
requested product:
    - same_product
    - same_product_different_size
    - related_product
    - different_variant

Deliberately never reduces the Google Shopping listings to one winner - a
shopping comparison is expected to be a genuine mix (see
`models.ShoppingComparisonResult`), and every classifiable offer is returned
so the UI can show the user real, comparable choices rather than a single
pipeline decision.

Flow:
    one Google Shopping search
    -> keep approved retailers only (`shopping_retailers.py`); every other
       seller is discarded here, with no fallback to unknown sellers
    -> keep the first `_MAX_CANDIDATES` of those, in Google's order
    -> resolve each one's direct retailer URL
       (one `google_immersive_product` call each)
    -> fetch bounded product text from that retailer page
    -> Nemotron classification
    -> ranking for display (`rank_comparison_offers`)
The retailer URL resolved before classification is the one the UI links
to; nothing is resolved a second time.

Architectural split (see `models.ShoppingMatchClassification`'s docstring
for the category definitions): SerpApi's job is DISCOVERY only (find
candidate listings); Nemotron's job is PRODUCT UNDERSTANDING (does this
candidate's title/price/seller/thumbnail actually match what the user
asked for, semantically, not by keyword overlap); this stage's job is
COMPARISON (group by classification, sort, resolve links) - Nemotron is
never asked to sort by price or pick a "winner". `app.py` shows primary
"Comparable offers" (same_product / same_product_different_size) and a
secondary "Related alternatives" section (related_product).
different_variant offers are kept in the result but only counted in the
UI ("N unrelated listings hidden"), never shown as rows.

Amazon is NOT part of this flow. It was originally used here as an optional
"canonical/verified reference" step, but that made this stage slower (an
extra multi-candidate search+verify+gallery-download loop), far more
expensive (up to 6 extra SerpApi credits per query - see the live credit
comparison this was decided from), and added no capability this stage
actually needs: Google Shopping's own listing titles are already sufficient
context for classification (`verify_product_match`/`classify_shopping_offers`
were already proven to work correctly with no canonical title - that was
already the code path whenever Amazon found nothing). The existing Amazon
SerpApi module (`search/retailers/amazon_serpapi.py`) is untouched and
remains fully in use by the existing image/gallery retrieval pipeline
(`retailer_gallery_stage.py`/`selection_stage.py`, reached from the
"Single product" UI tab and `run.py`) - this file simply no longer imports
or calls it.

Evidence for classification (see `_classify_offers`): text is primary -
the listing title plus, when it can be retrieved, bounded text from the
candidate's retailer page (`product_page_text.py`); the thumbnail is
supplemental. None of it is required: if the retailer URL can't be
resolved, the page can't be read, or the thumbnail can't be downloaded,
the candidate is still classified from whatever is left.

Failure handling principle: a classification failure is NOT evidence of a
product mismatch - it is an infrastructure failure, not a product
judgment. An offer whose batch couldn't be classified is preserved in
`ShoppingComparisonResult.unclassified_offers`, never silently dropped and
never treated as a match. The UI reports these as a count ("N offers
couldn't be verified") rather than showing unverified rows.

Cost: SerpApi usage per comparison is 1 search + 1 `google_immersive_product`
call per candidate, capped at `_MAX_CANDIDATES` candidates (see
`_resolve_retailer_urls`).
Retailer pages and thumbnails are fetched directly, not through SerpApi.
Thumbnails go to a private, per-run OS temp directory for classification
only - never persisted under `images/`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Final

from classifier.nemotron import NemotronVisionModel
from interfaces import ImageDownloader
from models import (
    ClassifiedShoppingOffer,
    DownloadedImage,
    ImageCandidate,
    SearchIntent,
    SearchQuery,
    ShoppingComparisonResult,
    ShoppingMatchClassification,
    ShoppingMatchVerdict,
    ShoppingOffer,
)
from product_page_text import fetch_product_page_text, is_product_page_url
from search.query_normalizer import normalize_for_search
from search.retailers import google_shopping_serpapi
from shopping_retailers import canonical_retailer, normalize_source
from storage.local_filesystem import LocalFilesystemStorage

logger = logging.getLogger(__name__)

# Upper bound on how many results to read from the single Google Shopping
# call. SerpApi returns one fixed page of ~40 (`start`/`num` have no effect),
# so this reads the whole page; it is also the most candidates that can
# ever reach Nemotron. No extra API calls.
_MAX_RAW_SHOPPING_RESULTS: Final[int] = 100

# How many approved candidates, in Google's order, go through the expensive
# part of the pipeline: a `google_immersive_product` call for the retailer
# URL, a fetch of that retailer page, and Nemotron classification. This
# bounds SerpApi usage at 1 search + this many calls per comparison, and
# Nemotron work at two batches. Live runs classifying every approved
# candidate (25-31 of them) took 2-2.5 minutes and hit NVIDIA rate limits.
# Approved candidates beyond the cap are not processed or shown.
_MAX_CANDIDATES: Final[int] = 10

# NemotronVisionModel.classify_shopping_offers's own hard cap
# (`_MAX_IMAGES_PER_REQUEST`) - offers beyond this are classified in
# multiple batched requests, not dropped.
_MAX_OFFERS_PER_CLASSIFICATION_BATCH: Final[int] = 5

# A failed classification batch is tried this many times in total. The pause
# before retry N is N x this delay (5s, then 10s) - see `_classify_batch`.
_BATCH_ATTEMPTS: Final[int] = 3
_BATCH_RETRY_DELAY_SECONDS: Final[float] = 5.0


def _build_offer_label(number: int, offer: ShoppingOffer, page_text: str | None) -> str:
    """
    One candidate's text evidence for Nemotron: listing details plus
    product-page text when available. The URL is shown only when it is the
    resolved retailer page - Google's own listing link repeats the search
    query and is not shown to the model.
    """
    link = str(offer.product_link)
    return (
        f"{number}. Title: {offer.title}\n"
        f"   Retailer: {offer.source or 'unknown seller'}\n"
        f"   Price: {offer.price or 'price unknown'}\n"
        f"   URL: {link if is_product_page_url(link) else 'not available'}\n"
        f"   Page text: {page_text or 'not available'}"
    )


# A listing whose title says it is the container, empty packaging, or an
# accessory rather than the product itself: "empty bottle", "bottle only",
# "cap only", "case for <product>", "accessories". Deliberately narrow - a
# bare "bottle", "jar", "pouch" or "pack" is how real products are packaged
# and never matches on its own.
_CONTAINER_WORDS: Final[str] = (
    r"(?:bottle|jar|container|box|tin|can|pouch|packet|packaging|dispenser|tube|cap|lid|case|cover)s?"
)
_CONTAINER_OR_ACCESSORY_MARKERS: Final[re.Pattern[str]] = re.compile(
    rf"\bempty\b(?:\W+\w+){{0,2}}?\W+{_CONTAINER_WORDS}\b"
    rf"|\b{_CONTAINER_WORDS}\W+only\b"
    r"|\b(?:case|cover|holder|stand|protector|guard|skin|sleeve)\W+for\b"
    r"|\baccessor(?:y|ies)\b",
    re.IGNORECASE,
)

# A size, weight, storage or pack quantity stated in the user's query:
# "1 kg", "950g", "500 ml", "256GB", "pack of 2", "12 pcs", "6 x 70".
_STATED_SIZE: Final[re.Pattern[str]] = re.compile(
    r"\d+(?:\.\d+)?\s*(?:kgs?|gms?|grams?|g|mg|ml|ltrs?|litres?|liters?|l|oz|lbs?|gb|tb|inch(?:es)?|cm|mm)\b"
    r"|\bpack\s+of\s+\d+\b"
    r"|\b\d+\s*(?:pcs?|pieces?|units?|count|packs?|sachets?|tablets?|capsules?)\b"
    r"|\b\d+\s*x\s*\d+\b",
    re.IGNORECASE,
)


def _query_states_size(query: str) -> bool:
    """Whether the user's query names a size, weight, storage, or pack quantity."""
    return _STATED_SIZE.search(query) is not None


def _is_container_or_accessory_listing(title: str, query: str) -> bool:
    """
    Whether `title` describes a container, empty packaging, or an accessory
    for a product rather than the product itself - unless the user's own
    `query` asks for exactly that kind of item.
    """
    if _CONTAINER_OR_ACCESSORY_MARKERS.search(query):
        return False
    return _CONTAINER_OR_ACCESSORY_MARKERS.search(title) is not None


def _reconcile_verdict(
    verdict: ShoppingMatchVerdict, offer: ShoppingOffer, query: str
) -> ShoppingMatchVerdict:
    """
    Deterministic consistency rules applied to a model verdict. They never
    make an offer relevant; they only correct verdicts that contradict the
    listing, the query, or the model's own answers:

    1. Container/accessory: a listing whose title says it is the container,
       empty packaging, or an accessory (see `_is_container_or_accessory_listing`)
       is not the product, so it cannot be relevant -> DIFFERENT_VARIANT. In
       live runs the model called an empty ketchup bottle "related" to, and
       once the "same product" as, a ketchup query.
    2. Self-contradiction: every relevant category needs the brand or the
       kind of product to match, so a candidate the model itself marks as
       matching NEITHER (`brand_match` and `product_match` both False)
       -> DIFFERENT_VARIANT.
    3. Size: when the query states no size, a candidate cannot be "the same
       product in a different size" -> SAME_PRODUCT.

    The category set is unchanged; the model's reason and evidence are kept.
    """
    classification = verdict.classification
    if classification in _COMPARISON_RANK:
        if _is_container_or_accessory_listing(offer.title, query):
            logger.info(
                "[Shopping comparison] %r is a container/accessory listing, not the product - "
                "treating the model's %s verdict as not relevant",
                offer.title[:80],
                classification.value,
            )
            return verdict.model_copy(
                update={"classification": ShoppingMatchClassification.DIFFERENT_VARIANT}
            )
        if verdict.brand_match is False and verdict.product_match is False:
            logger.info(
                "[Shopping comparison] verdict %s contradicts the model's own brand_match=False and "
                "product_match=False - treating as not relevant (reason given: %r)",
                classification.value,
                verdict.reason,
            )
            return verdict.model_copy(
                update={"classification": ShoppingMatchClassification.DIFFERENT_VARIANT}
            )

    if (
        classification == ShoppingMatchClassification.SAME_PRODUCT_DIFFERENT_SIZE
        and not _query_states_size(query)
    ):
        return verdict.model_copy(
            update={"classification": ShoppingMatchClassification.SAME_PRODUCT}
        )

    return verdict


def _keep_approved_retailers(offers: list[ShoppingOffer]) -> list[ShoppingOffer]:
    """
    Only offers sold by an approved retailer (`shopping_retailers.APPROVED_RETAILERS`),
    in Google's original order. Every other seller - including offers with
    no seller name - is discarded; there is no fallback to unknown sellers.
    Nothing is judged for relevance here - that stays Nemotron's job.
    """
    return [offer for offer in offers if canonical_retailer(offer.source) is not None]


# Order of the consumer-facing comparison list. different_variant is absent
# on purpose: it is not relevant and never enters the list.
_COMPARISON_RANK: Final[dict[ShoppingMatchClassification, int]] = {
    ShoppingMatchClassification.SAME_PRODUCT: 0,
    ShoppingMatchClassification.SAME_PRODUCT_DIFFERENT_SIZE: 1,
    ShoppingMatchClassification.RELATED_PRODUCT: 2,
}

_TITLE_NON_WORD_RUNS = re.compile(r"[\W_]+")


def _comparison_sort_key(classified: ClassifiedShoppingOffer) -> tuple[int, bool, float, bool, int]:
    """Classification rank, then price ascending (missing last), then Google position (missing last)."""
    offer = classified.offer
    price = offer.extracted_price
    position = offer.position
    return (
        _COMPARISON_RANK[classified.verdict.classification],
        price is None,
        price if price is not None else 0.0,
        position is None,
        position if position is not None else 0,
    )


def _duplicate_key(
    classified: ClassifiedShoppingOffer,
) -> tuple[str, ShoppingMatchClassification, str]:
    """Same retailer + same classification + same normalized title = the same listing."""
    offer = classified.offer
    retailer = canonical_retailer(offer.source) or normalize_source(offer.source or "")
    title = _TITLE_NON_WORD_RUNS.sub(" ", offer.title.casefold()).strip()
    return retailer, classified.verdict.classification, title


def rank_comparison_offers(offers: list[ClassifiedShoppingOffer]) -> list[ClassifiedShoppingOffer]:
    """
    The consumer-facing comparison list, in a deterministic order:
    same_product, then same_product_different_size, then related_product;
    within each, cheapest first, then Google's original position. Python's
    stable sort keeps input order when both are equal or missing.

    different_variant offers are excluded (unclassified offers never get
    here - they aren't in `ShoppingComparisonResult.offers`). When one
    retailer lists the same product twice under the same classification,
    only the cheapest is kept; different sizes/configurations have different
    titles and are all kept. Pure function - the input list is not modified.
    """
    relevant = [c for c in offers if c.verdict.classification in _COMPARISON_RANK]
    ranked: list[ClassifiedShoppingOffer] = []
    seen: set[tuple[str, ShoppingMatchClassification, str]] = set()
    for classified in sorted(relevant, key=_comparison_sort_key):
        key = _duplicate_key(classified)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(classified)
    return ranked


class ShoppingComparisonStage:
    """Runs the Google-Shopping-search -> classification workflow (no Amazon step - see module docstring)."""

    def __init__(
        self,
        downloader: ImageDownloader,
        vision_model: NemotronVisionModel,
        storage: LocalFilesystemStorage,
        *,
        max_candidates: int = _MAX_CANDIDATES,
    ) -> None:
        self._downloader = downloader
        self._vision_model = vision_model
        # Unused by this stage today (there's nothing left to persist once
        # Amazon's gallery download is gone) - kept as a constructor
        # parameter for symmetry with `run.build_shopping_stage()` and in
        # case a future need to persist something here reappears; storing a
        # reference costs nothing.
        self._storage = storage
        self._max_candidates = max_candidates

    async def compare(self, product_name: str) -> ShoppingComparisonResult:
        """
        Build a `ShoppingComparisonResult` for `product_name`. Never raises
        for "nothing found" cases (empty `offers`/`unclassified_offers` are
        valid, reportable outcomes, not exceptions) - only genuinely
        unexpected failures propagate.
        """
        logger.info("Starting shopping comparison for product=%r", product_name)
        search_query_text = normalize_for_search(product_name)
        if search_query_text != product_name:
            logger.info("Normalized search query: %r -> %r", product_name, search_query_text)

        offers, search_failed = await self._search_shopping_offers(search_query_text)
        logger.info("Found %d shopping offer(s) for product=%r", len(offers), product_name)

        candidates = _keep_approved_retailers(offers)
        discarded_sellers = sorted(
            {
                offer.source or "(no seller name)"
                for offer in offers
                if canonical_retailer(offer.source) is None
            }
        )
        logger.info(
            "Kept %d of %d offer(s) from approved retailers; discarded sellers: %s",
            len(candidates),
            len(offers),
            discarded_sellers,
        )

        if len(candidates) > self._max_candidates:
            logger.info(
                "Processing the first %d of %d approved candidate(s) (cap)",
                self._max_candidates,
                len(candidates),
            )
            candidates = candidates[: self._max_candidates]

        candidates = await self._resolve_retailer_urls(candidates)
        classified, unclassified = await self._classify_offers(candidates, product_name)

        return ShoppingComparisonResult(
            product_name=product_name,
            offers=classified,
            unclassified_offers=unclassified,
            search_failed=search_failed,
        )

    async def _search_shopping_offers(
        self, search_query_text: str
    ) -> tuple[list[ShoppingOffer], bool]:
        """
        Wraps `google_shopping_serpapi.search_shopping_offers` with the same
        skip-and-continue resilience used throughout the rest of this
        codebase (e.g. `selection_stage.py`'s `_search_download_select`) - a
        transient SerpApi failure here degrades to "no shopping offers
        found" for this run rather than crashing the whole comparison.

        Returns `(offers, search_failed)`: `search_failed=True` only when
        the search call itself raised - never when it genuinely succeeded
        with zero results. This distinction is preserved all the way to
        `ShoppingComparisonResult.search_failed` so the UI never represents
        a failed search as a legitimate empty result (see that field's
        docstring in `models.py`).
        """
        try:
            offers = await google_shopping_serpapi.search_shopping_offers(
                search_query_text, _MAX_RAW_SHOPPING_RESULTS
            )
            return offers, False
        except Exception as exc:  # noqa: BLE001 - skip-and-continue, but flag the failure (not silent)
            logger.warning("[Shopping comparison] Google Shopping search failed: %s", exc)
            return [], True

    async def _classify_offers(
        self, offers: list[ShoppingOffer], product_name: str
    ) -> tuple[list[ClassifiedShoppingOffer], list[ShoppingOffer]]:
        """
        Gather each offer's evidence - product-page text (primary) and
        thumbnail (supplemental, downloaded into a private per-run temp
        directory, never persisted) - then classify in batches of at most
        `_MAX_OFFERS_PER_CLASSIFICATION_BATCH`, run concurrently, and return
        `(classified, unclassified)` in the original candidate order.

        Neither kind of evidence is required: with no page text the model
        judges from title/retailer/price/image, and with no thumbnail it
        judges from the text alone. An offer lands in `unclassified` -
        never dropped - only when its batch could not be classified (see
        `_classify_batch`), which is not evidence the product doesn't match.
        A successful batch's offers are still returned as `classified` even
        if a different batch in the same run failed.
        """
        if not offers:
            return [], []

        query = SearchQuery(text=product_name, intent=SearchIntent.GENERIC)

        with tempfile.TemporaryDirectory(prefix="pir_shopping_offers_") as tmp_dir:
            destination_dir = Path(tmp_dir)
            downloaded, page_texts = await asyncio.gather(
                asyncio.gather(
                    *(
                        self._download_offer_thumbnail(offer, query, destination_dir)
                        for offer in offers
                    )
                ),
                asyncio.gather(
                    *(
                        fetch_product_page_text(str(offer.product_link), about=offer.title)
                        for offer in offers
                    )
                ),
            )
            candidates = list(zip(offers, downloaded, page_texts))
            batches = [
                candidates[start : start + _MAX_OFFERS_PER_CLASSIFICATION_BATCH]
                for start in range(0, len(candidates), _MAX_OFFERS_PER_CLASSIFICATION_BATCH)
            ]
            outcomes = await asyncio.gather(
                *(
                    self._classify_batch(batch_number, batch, product_name)
                    for batch_number, batch in enumerate(batches, start=1)
                )
            )

        classified: list[ClassifiedShoppingOffer] = []
        unclassified: list[ShoppingOffer] = []
        for batch, verdicts in zip(batches, outcomes):
            if verdicts is None:
                unclassified.extend(offer for offer, _, _ in batch)
            else:
                classified.extend(
                    ClassifiedShoppingOffer(offer=offer, verdict=verdict)
                    for (offer, _, _), verdict in zip(batch, verdicts)
                )
        return classified, unclassified

    async def _classify_batch(
        self,
        batch_number: int,
        batch: list[tuple[ShoppingOffer, DownloadedImage | None, str | None]],
        product_name: str,
    ) -> list[ShoppingMatchVerdict] | None:
        """
        Classify one batch, or return `None` if it could not be classified.
        NVIDIA Build's free endpoint intermittently refuses requests (HTTP
        503 "request limit reached") even after the client's own quick
        retries, and losing a batch loses half of a typical comparison - so
        a failed batch is tried again after a pause, up to `_BATCH_ATTEMPTS`
        times in total, waiting longer each time.
        """
        image_paths = [image.local_path if image is not None else None for _, image, _ in batch]
        offer_labels = [
            _build_offer_label(number, offer, page_text)
            for number, (offer, _, page_text) in enumerate(batch, start=1)
        ]
        logger.info(
            "[Shopping comparison] classifying batch %d: %d candidate(s), %d with retailer page text, %d with an image",
            batch_number,
            len(batch),
            sum(1 for _, _, page_text in batch if page_text),
            sum(1 for _, image, _ in batch if image is not None),
        )

        for attempt in range(1, _BATCH_ATTEMPTS + 1):
            try:
                verdicts = await self._vision_model.classify_shopping_offers(
                    image_paths, product_name, offer_labels
                )
                verdicts = [
                    _reconcile_verdict(verdict, offer, product_name)
                    for (offer, _, _), verdict in zip(batch, verdicts)
                ]
                break
            except Exception as exc:  # noqa: BLE001 - preserve as unclassified, not a mismatch
                if attempt < _BATCH_ATTEMPTS:
                    delay = _BATCH_RETRY_DELAY_SECONDS * attempt
                    logger.warning(
                        "[Shopping comparison] classification failed for a batch of %d offer(s) - "
                        "retrying in %.0fs: %s",
                        len(batch),
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning(
                    "[Shopping comparison] classification failed for a batch of %d offer(s) - "
                    "preserving as unclassified rather than dropping (a failure here is not "
                    "evidence of a mismatch): %s",
                    len(batch),
                    exc,
                )
                return None

        return verdicts

    async def _resolve_retailer_urls(self, offers: list[ShoppingOffer]) -> list[ShoppingOffer]:
        """
        Find each approved candidate's direct retailer URL BEFORE
        classification, so the retailer's product page can be read as
        evidence - see `google_shopping_serpapi.resolve_direct_merchant_link`.
        A resolved URL replaces `offer.product_link`, and that same URL is
        what the UI later links to: nothing is resolved again after
        classification.

        COST: one `engine=google_immersive_product` SerpApi call per distinct
        (page token, seller) among `offers`. Callers must pass the capped,
        approved-retailer candidates only - unknown sellers are never
        resolved. Candidates sharing a token and seller share one call.

        Never drops a candidate: with no `page_token`, no matching store, or
        a failed request, the offer simply keeps Google's listing link and
        is classified from its title/retailer/price/image.
        """
        indices_by_key: dict[tuple[str, str], list[int]] = {}
        for index, offer in enumerate(offers):
            if offer.page_token:
                key = (offer.page_token, normalize_source(offer.source or ""))
                indices_by_key.setdefault(key, []).append(index)
        if not indices_by_key:
            return offers

        # `return_exceptions=True` is defense in depth: `resolve_direct_merchant_link`
        # never raises by contract, but one unexpected error must not lose the whole comparison.
        resolved_links = await asyncio.gather(
            *(
                google_shopping_serpapi.resolve_direct_merchant_link(
                    page_token, offers[indices[0]].source
                )
                for (page_token, _), indices in indices_by_key.items()
            ),
            return_exceptions=True,
        )

        updated = list(offers)
        resolved_count = 0
        for indices, link in zip(indices_by_key.values(), resolved_links):
            if isinstance(link, Exception):
                logger.warning(
                    "[Shopping comparison] retailer-URL resolution raised unexpectedly for offer=%r "
                    "(keeping Google's link): %s",
                    offers[indices[0]].title,
                    link,
                )
                continue
            if link:
                for index in indices:
                    updated[index] = offers[index].model_copy(update={"product_link": link})
                resolved_count += len(indices)

        logger.info(
            "[Shopping comparison] resolved retailer URLs for %d of %d candidate(s) "
            "using %d google_immersive_product call(s)",
            resolved_count,
            len(offers),
            len(indices_by_key),
        )
        return updated

    async def _download_offer_thumbnail(
        self, offer: ShoppingOffer, query: SearchQuery, destination_dir: Path
    ) -> DownloadedImage | None:
        candidate = ImageCandidate(
            url=offer.thumbnail_url,
            source_provider="google_shopping",
            source_query=query,
            page_url=offer.product_link,
        )
        try:
            return await self._downloader.download(candidate, destination_dir=destination_dir)
        except Exception as exc:  # noqa: BLE001 - the image is supplemental; classify from text instead
            logger.warning(
                "[Shopping comparison] could not download thumbnail (title=%r) url=%s - classifying "
                "from text only: %s",
                offer.title,
                offer.thumbnail_url,
                exc,
            )
            return None
