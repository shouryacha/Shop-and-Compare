"""
Retailer-first image retrieval stage.

For each retailer, in priority order (Amazon -> Google Shopping):
    1. Search for the product via SerpApi (`engine=amazon` for Amazon,
       `engine=google_shopping` for Google Shopping) - no browser involved
       for either; `search/retailers/<name>_serpapi.py`.
    2. Extract that product's full gallery, also via SerpApi
       (`engine=amazon_product` keyed by the ASIN parsed from the product
       URL, or `engine=google_immersive_product` keyed by the page token
       embedded in the product URL) and read `product_results.thumbnails[]`.
    3. Download ONLY the first gallery image (reusing the existing
       `ImageDownloader.download()`) into a private, per-retailer-attempt OS
       temp directory - not anywhere under `images/`. This is purely a
       scratch location for the verification step below; it is always
       removed once the attempt finishes, whether accepted or rejected.
    4. Ask the existing vision-model client whether that one image is
       genuinely the requested product (`NemotronVisionModel.verify_product_match`) -
       tolerant of retailer wording differences, strict about a genuinely
       different product family/variant. This step always receives the
       ORIGINAL, unmodified product name, plus the retailer's own listing
       title when the extractor found one (`GalleryResult.product_title`)
       as additional textual evidence alongside the image - added after a
       live false-positive where a genuinely different product variant was
       accepted as just a naming-convention difference (see
       `prompts.py`'s product-match system prompt). `verify_product_match`
       retries transient NVIDIA Build failures (HTTP 500s, malformed/missing/
       empty responses, timeouts) a small number of times before giving up
       (see `classifier/nemotron.py`) - an API failure is never treated as a
       rejection.
    5. If accepted: download every remaining gallery image, preserve the
       retailer's exact order, and persist all of them - including the
       first, already-downloaded image - directly into the final
       per-product folder as `image_01.<ext>, image_02.<ext>, ...`
       (reusing the existing `LocalFilesystemStorage`, extended with
       `persist_gallery`). No front/back/side judgement is made anywhere in
       this path - the whole gallery is kept as-is. There is no separate
       "downloads" tree left behind: the temp directory is removed right
       after `persist_gallery` copies the accepted images out of it.
       If rejected, or any step fails: the temp directory (and whatever was
       downloaded into it) is discarded, and the next retailer is tried.
    6. If neither Amazon nor Google Shopping produces an accepted result:
       fall back to the existing `selection_stage.ProductImageSelectionStage`
       (the Google Images comparative-selection workflow via SerpApi),
       unchanged.

Both active retailers are SerpApi-backed and never touch the `page` argument
threaded through `_try_retailer`/`_try_candidate` (kept only for contract
parity with the dormant Playwright-based modules below) - so `run()` no
longer launches a Playwright browser at all; `page` is passed through as
`None`.

Flipkart and Zepto were removed from the active registry (they required
Playwright and their own native-search mechanics); `search/retailers/
flipkart.py` and `search/retailers/zepto.py` are left in place, unreferenced,
as an easy rollback - same treatment already given to the old Playwright-based
`search/retailers/amazon.py`.

Depends concretely on `classifier.nemotron.NemotronVisionModel` (for
`verify_product_match`) and `storage.local_filesystem.LocalFilesystemStorage`
(for `persist_gallery`) rather than on the generic `interfaces.StorageProvider`
- both capabilities are specific to those concrete adapters, following the
same pattern already used for `select_best_candidate` in
`selection_stage.py`. `ImageDownloader` stays interface-typed and fully
reused as-is.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from classifier.nemotron import NemotronVisionModel
from interfaces import ImageDownloader
from models import DownloadedImage, ImageCandidate, ProductImageResult, ProductMatchVerdict, SearchIntent, SearchQuery
from search.query_normalizer import normalize_for_search
from search.retailers import amazon_serpapi, google_shopping_serpapi
from selection_stage import ProductImageSelectionStage
from storage.local_filesystem import LocalFilesystemStorage
from verification_overrides import OverrideInfo, apply_deterministic_checks

logger = logging.getLogger(__name__)

# Priority order: Amazon -> Google Shopping -> Google Images (fallback).
# Flipkart and Zepto were removed from this active registry (Playwright-
# based, replaced by an all-SerpApi retailer lineup) - their modules are
# left in place, unreferenced, as an easy rollback (see module docstring).
# Blinkit and BigBasket remain removed entirely (never fully working in this
# sandbox). Both retailers here are SerpApi-backed and ignore the `page`
# argument entirely - it's kept only for contract parity with the dormant
# Playwright-based modules. The old Playwright-based Amazon module
# (`search/retailers/amazon.py`) is likewise left in place, unreferenced, as
# an easy rollback if needed.
_RETAILERS: list[dict[str, Any]] = [
    {"name": "Amazon", "search_fn": amazon_serpapi.search_product_page, "extractor": amazon_serpapi.extract_gallery},
    {
        "name": "Google Shopping",
        "search_fn": google_shopping_serpapi.search_product_page,
        "extractor": google_shopping_serpapi.extract_gallery,
    },
]

VERIFICATION_CONFIDENCE_THRESHOLD = 0.6

# Debugging switch (added to isolate the retailer-first pipeline for a
# manual 99-product regression pass): when False, `run()` never calls
# `fallback_stage.run()` - if Amazon and Google Shopping both fail, it
# returns a clearly-marked "no result" `ProductImageResult` instead.
# Re-enabled (True) for normal operation: Amazon -> Google Shopping ->
# Google Images. Flip back to False for a retailer-only debugging pass;
# nothing else in this file changes either way - the fallback
# implementation itself (selection_stage.py) is untouched regardless.
ENABLE_GOOGLE_FALLBACK = True

# How many of a retailer's own top search results to try, in order, before
# giving up on that retailer and moving to the next one. Rejecting the
# first candidate no longer means abandoning the retailer entirely - the
# correct product sometimes ranks below a bigger/more prominent competitor
# for the same generic category (see HANDOFF.md's brand-mismatch findings).
# Tune this down (e.g. to 3) if the added latency/vision-API cost on
# hard-to-match products proves too high in practice.
_MAX_CANDIDATES_PER_RETAILER = 5


class RetailerGalleryStage:
    """Runs the retailer-first search -> extract -> verify -> full-gallery-download workflow."""

    def __init__(
        self,
        downloader: ImageDownloader,
        vision_model: NemotronVisionModel,
        storage: LocalFilesystemStorage,
        fallback_stage: ProductImageSelectionStage,
        *,
        min_match_confidence: float = VERIFICATION_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Args:
            fallback_stage: Run unchanged if no retailer produces an
                accepted result.
            min_match_confidence: Extra safety floor on top of
                `ProductMatchVerdict.is_match` - a retailer is only
                accepted if the model both said "match" AND its confidence
                clears this bar.
        """
        self._downloader = downloader
        self._vision_model = vision_model
        self._storage = storage
        self._fallback_stage = fallback_stage
        self._min_match_confidence = min_match_confidence

    async def run(self, product_name: str, erp_brand: str | None = None) -> ProductImageResult:
        """
        Try every retailer in priority order; fall back to
        `fallback_stage.run()` if none of them produce an accepted result.

        `erp_brand`: the ERP source's own brand field (e.g. an Excel "Brand"
        column), if the caller has one - `None` by default (the CLI and
        single-product UI have no such field to pass). When given, it's
        used ONLY for the deterministic post-verification brand check (see
        `verification_overrides.py`) - never sent to the vision model or
        used for search.
        """
        logger.info("Starting retailer-first retrieval for product=%r", product_name)

        # Sent to every retailer's OWN search box in place of the raw ERP
        # name - strips internal packaging/inventory codes (PP, PCH, PET,
        # BTL, ...) and expands known abbreviations (see
        # `search.query_normalizer`). Verification always uses the
        # original, unmodified `product_name` below, never this.
        search_query_text = normalize_for_search(product_name)
        if search_query_text != product_name:
            logger.info("Normalized search query: %r -> %r", product_name, search_query_text)

        # Neither active retailer touches `page` (both are SerpApi-backed),
        # so there's no Playwright browser to launch here anymore - see
        # module docstring. `page` is still threaded through for contract
        # parity with the dormant Playwright-based retailer modules.
        for retailer in _RETAILERS:
            result = await self._try_retailer(None, retailer, search_query_text, product_name, erp_brand)
            if result is not None:
                return result

        if not ENABLE_GOOGLE_FALLBACK:
            logger.info(
                "Google/SerpAPI fallback disabled.\n"
                "Retailer-first retrieval failed after Amazon and Google Shopping.\n"
                "Returning no result."
            )
            return ProductImageResult(
                product_name=product_name,
                metadata={
                    "retailer": None,
                    "fallback_disabled": True,
                    "status": "no_retailer_match_fallback_disabled",
                },
            )

        logger.info(
            "No retailer produced an accepted product page for %r; falling back to "
            "the Google Images pipeline",
            product_name,
        )
        return await self._fallback_stage.run(product_name)

    async def _try_retailer(
        self,
        page: Page | None,
        retailer: dict[str, Any],
        search_query_text: str,
        product_name: str,
        erp_brand: str | None,
    ) -> ProductImageResult | None:
        """
        Attempt one retailer end to end; return `None` once every candidate
        has failed or been rejected. `search_query_text` (normalized) is
        what's sent to the retailer's own search box; `product_name`
        (original, unmodified ERP name) is what verification and
        persistence use throughout - never the normalized form.

        Tries up to `_MAX_CANDIDATES_PER_RETAILER` of the retailer's own
        search results in order, not just the first: if a candidate is
        rejected (or fails at any step), the next candidate from the SAME
        retailer is tried before giving up on this retailer entirely. This
        matters because the correct product sometimes isn't the retailer's
        top result (e.g. a smaller/regional brand ranked below bigger
        competitors for the same generic category).
        """
        name = retailer["name"]
        search_fn = retailer["search_fn"]
        logger.info(
            "[%s] search_fn bound to %s.%s (module file: %s), invoking with search_query_text=%r",
            name,
            search_fn.__module__,
            search_fn.__qualname__,
            getattr(sys.modules.get(search_fn.__module__), "__file__", "?"),
            search_query_text,
        )

        try:
            candidate_urls = await search_fn(page, search_query_text)
        except Exception as exc:  # noqa: BLE001 - try the next retailer
            logger.warning("[%s] native search failed: %s", name, exc)
            return None

        logger.info("[%s] search_fn returned %d candidate(s): %r", name, len(candidate_urls), candidate_urls)

        if not candidate_urls:
            logger.info("[%s] no product page found in search results", name)
            return None

        candidate_total = len(candidate_urls)
        for candidate_index, product_url in enumerate(candidate_urls[:_MAX_CANDIDATES_PER_RETAILER]):
            candidate_number = candidate_index + 1
            logger.info("[%s] trying candidate %d/%d: %s", name, candidate_number, candidate_total, product_url)
            result = await self._try_candidate(
                page, retailer, product_url, product_name, erp_brand, candidate_number, candidate_total
            )
            if result is not None:
                return result

        logger.info(
            "[%s] all %d candidate(s) exhausted without an accepted match",
            name,
            min(len(candidate_urls), _MAX_CANDIDATES_PER_RETAILER),
        )
        return None

    async def _try_candidate(
        self,
        page: Page | None,
        retailer: dict[str, Any],
        product_url: str,
        product_name: str,
        erp_brand: str | None,
        candidate_number: int,
        candidate_total: int,
    ) -> ProductImageResult | None:
        """
        Attempt one candidate product page: extract its gallery, verify its
        first image, and persist the full gallery if accepted. Returns
        `None` on any failure or rejection so the caller can move on to the
        next candidate (or the next retailer).
        """
        name = retailer["name"]

        try:
            gallery = await retailer["extractor"](page, product_url)
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            logger.warning("[%s] failed to open/extract product page %s: %s", name, product_url, exc)
            return None

        if not gallery.image_urls:
            logger.info("[%s] no gallery images found at %s (%s)", name, product_url, gallery.source_description)
            return None

        logger.info("[%s] found %d gallery image(s) (%s)", name, len(gallery.image_urls), gallery.source_description)

        query = SearchQuery(text=product_name, intent=SearchIntent.GENERIC)

        # Everything downloaded here is only a verification candidate until
        # this retailer is actually accepted below - it lives in a private,
        # per-attempt OS temp directory (never under `images/`) and is
        # always removed when this `with` block exits, whether accepted or
        # rejected.
        with tempfile.TemporaryDirectory(prefix="pir_retailer_") as tmp_dir:
            destination_dir = Path(tmp_dir)

            first_downloaded = await self._download_one(
                gallery.image_urls[0], name, query, product_url, destination_dir
            )
            if first_downloaded is None:
                logger.warning("[%s] could not download the first gallery image; skipping", name)
                return None

            try:
                match_verdict = await self._vision_model.verify_product_match(
                    first_downloaded.local_path, product_name, candidate_title=gallery.product_title
                )
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                logger.warning("[%s] product-match verification failed: %s", name, exc)
                return None

            # Deterministic guardrail applied AFTER the model's own verdict,
            # never inside the verifier itself - see verification_overrides.py.
            # Only ever tightens (True -> False); a no-op when erp_brand is
            # unavailable or the two brands aren't confidently different.
            match_verdict, override_info = apply_deterministic_checks(match_verdict, erp_brand)
            if override_info is not None:
                self._log_override(name, candidate_number, candidate_total, product_url, override_info)

            if not match_verdict.is_match or match_verdict.confidence < self._min_match_confidence:
                logger.info(
                    "[%s] rejected: is_match=%s confidence=%.2f reason=%r",
                    name,
                    match_verdict.is_match,
                    match_verdict.confidence,
                    match_verdict.reason,
                )
                return None

            logger.info(
                "[%s] accepted (confidence=%.2f) - downloading full gallery (%d image(s))",
                name,
                match_verdict.confidence,
                len(gallery.image_urls),
            )

            rest_downloaded = await self._download_rest(
                gallery.image_urls[1:], name, query, product_url, destination_dir
            )
            all_downloaded = [first_downloaded, *rest_downloaded]

            # persist_gallery copies these straight into the final
            # per-product folder before the temp directory above is removed.
            return await self._persist(
                product_name, name, product_url, all_downloaded, match_verdict, product_name
            )

    @staticmethod
    def _log_override(
        retailer_name: str,
        candidate_number: int,
        candidate_total: int,
        product_url: str,
        info: OverrideInfo,
    ) -> None:
        """
        Structured, multi-line log block for a fired deterministic override -
        deliberately separate from the normal `[retailer] rejected: ...`
        line so it's easy to grep for specifically (`[Brand Override]`) when
        debugging or auditing regression runs, without having to reconstruct
        it from the rejection reason text.
        """
        logger.info(
            "[Brand Override]\n"
            "Retailer: %s\n"
            "Candidate: %d/%d\n"
            "ERP Brand: %s\n"
            "Extracted Brand: %s\n"
            "Match Method: %s\n"
            "Model is_match: %s\n"
            "Confidence: %.2f\n"
            "Final Decision: Reject\n"
            "Reason: Deterministic brand mismatch override\n"
            "URL: %s",
            retailer_name,
            candidate_number,
            candidate_total,
            info.erp_brand,
            info.extracted_brand,
            info.match_method,
            info.original_is_match,
            info.original_confidence,
            product_url,
        )

    async def _download_one(
        self, url: str, retailer_name: str, query: SearchQuery, product_url: str, destination_dir: Path
    ) -> DownloadedImage | None:
        """Download a single gallery image URL, returning `None` on failure."""
        candidate = ImageCandidate(
            url=url, source_provider=retailer_name.lower(), source_query=query, page_url=product_url
        )
        try:
            return await self._downloader.download(candidate, destination_dir=destination_dir)
        except Exception as exc:  # noqa: BLE001 - skip-and-continue, matches selection_stage.py
            logger.warning("[%s] skipping gallery image url=%s: %s", retailer_name, url, exc)
            return None

    async def _download_rest(
        self,
        image_urls: list[str],
        retailer_name: str,
        query: SearchQuery,
        product_url: str,
        destination_dir: Path,
    ) -> list[DownloadedImage]:
        """Download every remaining gallery image concurrently, skipping individual failures."""
        results = await asyncio.gather(
            *(self._download_one(url, retailer_name, query, product_url, destination_dir) for url in image_urls)
        )
        return [d for d in results if d is not None]

    async def _persist(
        self,
        product_name: str,
        retailer_name: str,
        product_url: str,
        downloaded_images: list[DownloadedImage],
        match_verdict: ProductMatchVerdict,
        search_query_used: str,
    ) -> ProductImageResult:
        """Save the full gallery in order and write metadata.json with retailer/verification details."""
        persisted_paths = await self._storage.persist_gallery(downloaded_images, product_name)

        metadata: dict[str, Any] = {
            "retailer": retailer_name,
            "retailer_product_url": product_url,
            "search_query_used": search_query_used,
            "gallery_image_count": len(persisted_paths),
            "gallery_image_paths": [str(p) for p in persisted_paths],
            "verification_confidence": match_verdict.confidence,
            "verification_reason": match_verdict.reason,
            "brand": match_verdict.brand,
            "product_name": match_verdict.product_name,
            "weight": match_verdict.weight,
        }

        result = ProductImageResult(
            product_name=product_name,
            provider_used={
                "search": "native_retailer_search",
                "vision": self._vision_model.name,
                "retailer": retailer_name,
            },
            metadata=metadata,
        )
        await self._storage.persist_result(result)
        return result
