"""
Focused tests for `shopping_comparison_stage.py` after removing Amazon from
this flow. Mirrors `tests/test_selection_stage.py`'s fake-provider approach:
a fake `ImageDownloader`/storage/vision-model injected via the constructor,
plus `unittest.mock.patch` on the module-level `google_shopping_serpapi`
function this stage calls directly.

Covers exactly the properties the architecture change and the failure-
handling fix are meant to guarantee:
    - Amazon is never imported or called from this stage at all.
    - A successful classification batch works as before.
    - A failed batch's offers are preserved as `unclassified_offers`, never
      dropped - "classification failure != product mismatch".
    - One batch failing doesn't affect another batch's successful offers.
    - A thumbnail download failure also preserves the offer as unclassified.
    - Zero Google Shopping results leaves both `offers` and
      `unclassified_offers` empty AND `search_failed=False` (the only
      genuine "nothing found" case).
    - A Google Shopping search-call failure (network/API/SerpApi
      exception) also leaves both lists empty, but sets
      `search_failed=True` - distinguishable from the genuine empty case
      above, so a caller (the UI) never represents an infrastructure
      failure as "nothing found".
    - The whole result page is read in one search call, and only offers
      sold by an approved retailer enter the pipeline - all of them are
      classified, and unknown sellers are never used as a fallback.
    - Each approved candidate's retailer URL is resolved once, before
      classification (never for unknown sellers, never a second time), and
      its retailer page text is passed to Nemotron as primary evidence.
    - Failed resolution, failed page extraction and a missing thumbnail all
      fall back to the remaining evidence - none of them drops a candidate.
    - An offer missing `price` still gets classified, with the offer-label
      text falling back to a readable placeholder.
    - The visible comparison is ranked and de-duplicated deterministically.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch

import pytest

import product_page_text
from interfaces import ImageDownloader, StorageProvider
from models import (
    ClassifiedShoppingOffer,
    DownloadedImage,
    ImageCandidate,
    ProductImageResult,
    ShoppingMatchClassification,
    ShoppingMatchVerdict,
    ShoppingOffer,
)
import shopping_comparison_stage
from search.retailers import google_shopping_serpapi
from shopping_comparison_stage import (
    ShoppingComparisonStage,
    _keep_approved_retailers,
    rank_comparison_offers,
)

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


# The real resolver, captured before the fixture below replaces it, for the
# one test that drives it end to end against a mocked HTTP layer.
_REAL_RESOLVE = google_shopping_serpapi.resolve_direct_merchant_link


@pytest.fixture(autouse=True)
def _offline_defaults() -> Iterator[None]:
    """
    Defaults for every test here, so nothing reaches SerpApi or a retailer
    site unless a test opts in with its own patch:
      - the SerpApi module's only network boundary refuses to run (an
        unpatched resolution once sent real requests with the `.env` key);
      - retailer-URL resolution finds nothing;
      - retailer page-text fetching finds nothing.
    A test's own `patch(...)` of the same name takes precedence.
    """

    def _refuse(url: str) -> dict[str, Any]:
        raise AssertionError("tests must not call SerpApi")

    with (
        patch("search.retailers.google_shopping_serpapi._fetch_json", side_effect=_refuse),
        patch(
            "shopping_comparison_stage.google_shopping_serpapi.resolve_direct_merchant_link",
            return_value=None,
        ),
        patch("shopping_comparison_stage.fetch_product_page_text", return_value=None),
        patch(
            "shopping_comparison_stage._BATCH_RETRY_DELAY_SECONDS", 0.0
        ),  # no real pause between attempts
    ):
        yield


class _FakeDownloader(ImageDownloader):
    """Always succeeds, unless `fail_urls` says this exact URL should fail."""

    def __init__(self, fail_urls: set[str] | None = None) -> None:
        self._fail_urls = fail_urls or set()

    async def download(self, candidate: ImageCandidate, destination_dir: Path) -> DownloadedImage:
        if str(candidate.url) in self._fail_urls:
            raise RuntimeError(f"simulated download failure for {candidate.url}")
        return DownloadedImage(candidate=candidate, local_path=Path("/fake/downloaded.jpg"))


class _FakeStorage(StorageProvider):
    async def persist_image(
        self, downloaded: DownloadedImage, product_name: str, side: str
    ) -> Path:
        return Path(f"/fake/{product_name}/{side}.jpg")

    async def persist_result(self, result: ProductImageResult) -> None:
        return None


class _FakeVisionModel:
    """Stub covering only `classify_shopping_offers`."""

    name = "fake_vision_model"

    def __init__(self, classify_fn: Any = None) -> None:
        self._classify_fn = classify_fn
        self.classify_calls: list[list[str]] = []
        self.image_path_calls: list[list[Path | None]] = []
        self.product_names: list[str] = []

    async def classify_shopping_offers(
        self,
        image_paths: list[Path | None],
        product_name: str,
        offer_labels: list[str],
        canonical_title: str | None = None,
    ) -> list[ShoppingMatchVerdict]:
        self.classify_calls.append(offer_labels)
        self.image_path_calls.append(image_paths)
        self.product_names.append(product_name)
        if self._classify_fn is not None:
            verdicts: list[ShoppingMatchVerdict] = self._classify_fn(image_paths, offer_labels)
            return verdicts
        return [
            ShoppingMatchVerdict(
                classification=ShoppingMatchClassification.SAME_PRODUCT,
                confidence=0.9,
                reason="fake",
                model_name=self.name,
            )
            for _ in image_paths
        ]


def _offer(
    i: int,
    url: str = "https://example.com/thumb.jpg",
    page_token: str | None = "token-default",
    source: str | None = None,
    title: str | None = None,
    price: float = 10.0,
) -> ShoppingOffer:
    # Sold by an approved retailer by default, so the allowlist doesn't
    # interfere with tests about other behavior.
    return ShoppingOffer(
        position=i,
        title=title if title is not None else f"Offer {i}",
        source=source if source is not None else "Amazon.in",
        price=f"Rs {price:g}",
        extracted_price=price,
        thumbnail_url=url if "?" not in url else f"{url}&i={i}",
        product_link="https://www.google.com/shopping/somelisting",
        page_token=page_token,
    )


def _verdict(classification: ShoppingMatchClassification) -> ShoppingMatchVerdict:
    return ShoppingMatchVerdict(
        classification=classification, confidence=0.9, reason="fake", model_name="fake"
    )


def _stage(
    vision_model: _FakeVisionModel, downloader: ImageDownloader | None = None
) -> ShoppingComparisonStage:
    return ShoppingComparisonStage(
        downloader=downloader or _FakeDownloader(),
        vision_model=vision_model,  # type: ignore[arg-type]
        storage=_FakeStorage(),  # type: ignore[arg-type]
    )


# --- missing candidate information degrades gracefully ------------------------


def test_offer_with_missing_price_still_gets_classified() -> None:
    """
    Google Shopping doesn't always populate `price` - the offer-label text
    built for Nemotron must fall back to a readable placeholder instead of
    failing, and the offer must still be classified.
    """
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    sparse_offer = ShoppingOffer(
        position=1,
        title="Some Product",
        source="Zepto",
        price=None,
        extracted_price=None,
        thumbnail_url="https://example.com/thumb.jpg",
        product_link="https://www.google.com/shopping/somelisting",
        page_token=None,
    )

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=[sparse_offer],
    ):
        result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 1
    label = vision_model.classify_calls[0][0]
    assert "price unknown" in label
    assert "Zepto" in label


# --- Amazon must never be called from this stage ------------------------------


def test_compare_never_calls_amazon_serpapi() -> None:
    """
    `shopping_comparison_stage` must not even import `amazon_serpapi` - the
    whole point of this change is that Shop & Compare is independently
    usable through Google Shopping alone.
    """
    assert not hasattr(sys.modules["shopping_comparison_stage"], "amazon_serpapi")

    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with patch(
        "search.retailers.amazon_serpapi.search_product_page",
        side_effect=AssertionError(
            "amazon_serpapi.search_product_page must not be called by Shop & Compare"
        ),
    ):
        with patch(
            "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
            return_value=[_offer(1)],
        ):
            result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 1


# --- Google Shopping results processed directly, successful path -------------


def test_compare_classifies_offers_directly_from_google_shopping() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [_offer(1), _offer(2)]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product 1kg"))

    assert len(result.offers) == 2
    assert result.unclassified_offers == []
    assert all(
        o.verdict.classification == ShoppingMatchClassification.SAME_PRODUCT for o in result.offers
    )
    assert result.search_failed is False


def test_compare_returns_genuine_empty_state_when_google_shopping_returns_nothing() -> None:
    """A successful search with zero results is NOT a search failure."""
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers", return_value=[]
    ):
        result = _run(stage.compare("Some Product"))

    assert result.offers == []
    assert result.unclassified_offers == []
    assert result.search_failed is False


def test_compare_sets_search_failed_when_google_shopping_search_raises() -> None:
    """
    A search-call exception (network timeout, HTTP/API error, SerpApi
    exception, malformed response) must be distinguishable from a genuine
    zero-result search - both leave `offers`/`unclassified_offers` empty,
    but only a real failure sets `search_failed=True`. This is the field
    the UI branches on to avoid ever showing "No comparable shopping
    offers found" for an infrastructure failure.
    """

    async def raising_search(*args: Any, **kwargs: Any) -> list[ShoppingOffer]:
        raise RuntimeError("simulated SerpAPI timeout")

    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        side_effect=raising_search,
    ):
        result = _run(stage.compare("Some Product"))

    assert result.offers == []
    assert result.unclassified_offers == []
    assert result.search_failed is True


def test_search_failure_state_is_distinguishable_from_genuine_empty_state() -> None:
    """
    The two "both lists empty" outcomes must never collapse into the same
    result shape - that's the exact gap this fix closes. A caller (the UI)
    must be able to tell them apart from `ShoppingComparisonResult` alone,
    without inspecting logs or exception text.
    """

    async def raising_search(*args: Any, **kwargs: Any) -> list[ShoppingOffer]:
        raise RuntimeError("simulated network failure")

    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers", return_value=[]
    ):
        genuine_empty = _run(stage.compare("Some Product"))

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        side_effect=raising_search,
    ):
        failed_search = _run(stage.compare("Some Product"))

    assert genuine_empty.offers == failed_search.offers == []
    assert genuine_empty.unclassified_offers == failed_search.unclassified_offers == []
    assert genuine_empty.search_failed is False
    assert failed_search.search_failed is True


# --- classification failure != product mismatch: offers are preserved -------


def test_failed_classification_batch_preserves_offers_as_unclassified() -> None:
    def classify_fn(image_paths: list[Path], offer_labels: list[str]) -> list[ShoppingMatchVerdict]:
        raise RuntimeError("simulated NVIDIA Build failure after retries")

    vision_model = _FakeVisionModel(classify_fn=classify_fn)
    stage = _stage(vision_model)
    offers = [_offer(1), _offer(2)]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert result.offers == []
    assert len(result.unclassified_offers) == 2
    assert {o.title for o in result.unclassified_offers} == {"Offer 1", "Offer 2"}
    assert result.search_failed is False  # the search itself succeeded; only classification failed


def test_partial_batch_failure_keeps_successful_batch_classified() -> None:
    """7 offers -> batches of 5 + 2. The first batch fails every attempt, the second succeeds."""

    def classify_fn(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        if len(offer_labels) == 5:
            raise RuntimeError("simulated failure on the first batch")
        return [
            ShoppingMatchVerdict(
                classification=ShoppingMatchClassification.DIFFERENT_VARIANT,
                confidence=0.9,
                reason="fake",
                model_name="fake_vision_model",
            )
            for _ in image_paths
        ]

    vision_model = _FakeVisionModel(classify_fn=classify_fn)
    stage = _stage(vision_model)
    offers = [_offer(i, url=f"https://example.com/{i}.jpg") for i in range(7)]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert len(result.unclassified_offers) == 5  # first (failed) batch
    assert len(result.offers) == 2  # second (successful) batch
    assert all(
        o.verdict.classification == ShoppingMatchClassification.DIFFERENT_VARIANT
        for o in result.offers
    )


def test_failed_batch_is_retried_and_recovers() -> None:
    """NVIDIA's intermittent 503s must not cost half the comparison: one more attempt is made."""
    attempts = {"n": 0}

    def classify_fn(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("simulated HTTP 503: request limit reached")
        return [_verdict(ShoppingMatchClassification.SAME_PRODUCT) for _ in image_paths]

    vision_model = _FakeVisionModel(classify_fn=classify_fn)
    stage = _stage(vision_model)
    offers = [_offer(i, url=f"https://example.com/{i}.jpg") for i in range(3)]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert attempts["n"] == 2
    assert len(result.offers) == 3
    assert result.unclassified_offers == []


def test_batch_gives_up_after_three_attempts() -> None:
    attempts = {"n": 0}

    def classify_fn(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        attempts["n"] += 1
        raise RuntimeError("simulated persistent failure")

    vision_model = _FakeVisionModel(classify_fn=classify_fn)
    stage = _stage(vision_model)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=[_offer(1)],
    ):
        result = _run(stage.compare("Some Product"))

    assert attempts["n"] == 3
    assert result.offers == []
    assert len(result.unclassified_offers) == 1


def test_classified_offers_keep_candidate_order_across_concurrent_batches() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [_offer(i, url=f"https://example.com/{i}.jpg") for i in range(10)]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert [c.offer.title for c in result.offers] == [f"Offer {i}" for i in range(10)]
    assert len(vision_model.classify_calls) == 2


def test_thumbnail_download_failure_still_classifies_the_offer_from_text() -> None:
    """The image is supplemental: an offer without one goes to Nemotron with no image, not to unclassified."""
    vision_model = _FakeVisionModel()
    failing_offer = _offer(1, url="https://example.com/bad.jpg")
    good_offer = _offer(2, url="https://example.com/good.jpg")
    downloader = _FakeDownloader(fail_urls={"https://example.com/bad.jpg"})
    stage = _stage(vision_model, downloader=downloader)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=[failing_offer, good_offer],
    ):
        result = _run(stage.compare("Some Product"))

    assert [c.offer.title for c in result.offers] == ["Offer 1", "Offer 2"]
    assert result.unclassified_offers == []
    image_paths = vision_model.image_path_calls[0]
    assert image_paths[0] is None  # no image for the failed download
    assert image_paths[1] is not None
    assert result.search_failed is False


# --- batching above the 5-image-per-request cap -------------------------------


def test_compare_batches_more_than_five_offers_into_multiple_classify_calls() -> None:
    vision_model = _FakeVisionModel()
    offers = [_offer(i, url=f"https://example.com/{i}.jpg") for i in range(7)]
    stage = _stage(vision_model)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 7
    assert len(vision_model.classify_calls) == 2  # 5 + 2
    assert len(vision_model.classify_calls[0]) == 5
    assert len(vision_model.classify_calls[1]) == 2


# --- one search call, whole page -----------------------------------------------


def test_stage_reads_the_full_result_page_in_one_search_call() -> None:
    """The whole ~40-result page is read, in a single search call."""
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers", return_value=[]
    ) as mock_search:
        _run(stage.compare("Some Product"))

    mock_search.assert_called_once()
    assert mock_search.call_args[0][1] >= 40  # (query, max_results) positional call


def test_compare_processes_ten_candidates_in_two_batches() -> None:
    vision_model = _FakeVisionModel()
    offers = [_offer(i, url=f"https://example.com/{i}.jpg") for i in range(10)]
    stage = _stage(vision_model)

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 10
    assert len(vision_model.classify_calls) == 2  # 5 + 5
    assert all(len(batch) == 5 for batch in vision_model.classify_calls)


# --- approved-retailer allowlist ----------------------------------------------
# Only offers sold by an approved retailer enter the pipeline. Everything
# else is discarded before Nemotron - no fallback to unknown sellers - and
# every approved candidate is classified.


def _titles(offers: list[ShoppingOffer]) -> list[str]:
    return [offer.title for offer in offers]


def test_non_approved_sellers_are_discarded() -> None:
    offers = [
        _offer(0, source="icluster technologies"),
        _offer(1, source="Amazon.in"),
        _offer(2, source="Namo India Mart"),
        _offer(3, source="Croma"),
        _offer(4, source="Shopsy By Flipkart"),
        _offer(5, source="Apple Premium Reseller"),
        _offer(6, source="Tata CLiQ Fashion"),
    ]

    assert _titles(_keep_approved_retailers(offers)) == ["Offer 1", "Offer 3"]


def test_offers_without_a_seller_name_are_discarded() -> None:
    offer = _offer(0)
    offer.source = None

    assert _keep_approved_retailers([offer]) == []


def test_google_order_is_preserved_among_approved_retailers() -> None:
    offers = [
        _offer(0, source="Unknown A"),
        _offer(1, source="Flipkart"),
        _offer(2, source="Unknown B"),
        _offer(3, source="amazon.in"),
        _offer(4, source="Zepto"),
        _offer(5, source="Samsung"),
    ]

    assert _titles(_keep_approved_retailers(offers)) == ["Offer 1", "Offer 3", "Offer 4", "Offer 5"]


def test_one_approved_retailer_is_not_capped() -> None:
    offers = [_offer(i, source="Amazon.in") for i in range(8)]

    assert len(_keep_approved_retailers(offers)) == 8


def test_unknown_sellers_are_never_used_as_fallback() -> None:
    """Only 3 approved offers among 20 results: exactly those 3 reach Nemotron, nothing else fills in."""
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [
        _offer(i, url=f"https://example.com/{i}.jpg", source=f"Local Store {i}") for i in range(20)
    ]
    offers[7].source = "Amazon.in"
    offers[12].source = "Croma"
    offers[18].source = "Blinkit"

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert _titles([c.offer for c in result.offers]) == ["Offer 7", "Offer 12", "Offer 18"]
    assert sum(len(batch) for batch in vision_model.classify_calls) == 3
    assert result.unclassified_offers == []


def test_only_the_first_ten_approved_candidates_are_processed() -> None:
    """
    A full page of approved-retailer offers: only the first 10, in Google's
    order, are resolved and classified. Live runs classifying every approved
    candidate took 2+ minutes and hit NVIDIA rate limits.
    """
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [_offer(i, url=f"https://example.com/{i}.jpg") for i in range(40)]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        result = _run(stage.compare("Some Product"))

    assert sum(len(batch) for batch in vision_model.classify_calls) == 10
    assert len(vision_model.classify_calls) == 2
    assert [c.offer.title for c in result.offers] == [f"Offer {i}" for i in range(10)]


def test_unknown_seller_flood_is_discarded_entirely() -> None:
    """The live iPhone case: one unknown seller filling most of Google's top results."""
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    flood = [
        _offer(i, url=f"https://example.com/{i}.jpg", source="icluster technologies")
        for i in range(8)
    ]
    majors = [
        _offer(20 + i, url=f"https://example.com/{20 + i}.jpg", source=name)
        for i, name in enumerate(
            ["Apple", "Croma", "Reliance Digital", "Vijay Sales", "Flipkart", "Amazon.in"]
        )
    ]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=flood + majors,
    ):
        result = _run(stage.compare("iphone 17"))

    sources = [c.offer.source for c in result.offers]
    assert sources == ["Apple", "Croma", "Reliance Digital", "Vijay Sales", "Flipkart", "Amazon.in"]


def test_no_approved_retailers_gives_an_empty_result_without_any_model_or_link_calls() -> None:
    """
    Search succeeded but no approved retailer sells it: an empty result
    (which the UI shows as "No matching offers found"), not a search
    failure - and neither Nemotron nor direct-link resolution is called.
    """
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [_offer(i, source=f"Local Store {i}") for i in range(10)]

    with patch(
        "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers",
        return_value=offers,
    ):
        with patch(
            "shopping_comparison_stage.google_shopping_serpapi.resolve_direct_merchant_link"
        ) as mock_resolve:
            result = _run(stage.compare("Some Product"))

    assert result.offers == []
    assert result.unclassified_offers == []
    assert result.search_failed is False
    assert vision_model.classify_calls == []
    mock_resolve.assert_not_called()


# --- consumer-facing ranking (after Nemotron) ---------------------------------
# Nemotron only classifies; this deterministic Python step orders the
# comparison: same_product, same_product_different_size, related_product;
# cheapest first within each; Google position breaks ties.

_SP = ShoppingMatchClassification.SAME_PRODUCT
_SIZE = ShoppingMatchClassification.SAME_PRODUCT_DIFFERENT_SIZE
_REL = ShoppingMatchClassification.RELATED_PRODUCT
_DIFF = ShoppingMatchClassification.DIFFERENT_VARIANT


def _ranked(
    title: str,
    classification: ShoppingMatchClassification,
    price: float | None = None,
    position: int | None = None,
    source: str | None = "Some Store",
) -> ClassifiedShoppingOffer:
    offer = ShoppingOffer(
        position=position,
        title=title,
        source=source,
        price=None if price is None else f"Rs {price}",
        extracted_price=price,
        thumbnail_url="https://example.com/thumb.jpg",
        product_link="https://www.google.com/shopping/somelisting",
    )
    return ClassifiedShoppingOffer(offer=offer, verdict=_verdict(classification))


def _ranked_titles(offers: list[ClassifiedShoppingOffer]) -> list[str]:
    return [c.offer.title for c in rank_comparison_offers(offers)]


def test_ranking_orders_by_classification_before_price() -> None:
    offers = [
        _ranked("related, cheapest", _REL, price=10),
        _ranked("different size", _SIZE, price=50),
        _ranked("exact, most expensive", _SP, price=500),
    ]

    assert _ranked_titles(offers) == [
        "exact, most expensive",
        "different size",
        "related, cheapest",
    ]


def test_ranking_sorts_price_ascending_within_a_classification() -> None:
    offers = [
        _ranked("A", _SP, price=74, position=1),
        _ranked("B", _SP, price=28, position=2),
        _ranked("C", _SP, price=35, position=3),
    ]

    assert _ranked_titles(offers) == ["B", "C", "A"]


def test_ranking_breaks_price_ties_by_google_position() -> None:
    offers = [
        _ranked("pos 9", _SP, price=28, position=9),
        _ranked("pos 2", _SP, price=28, position=2),
        _ranked("pos 5", _SP, price=28, position=5),
    ]

    assert _ranked_titles(offers) == ["pos 2", "pos 5", "pos 9"]


def test_ranking_puts_missing_prices_last_and_keeps_order_when_all_equal() -> None:
    offers = [
        _ranked("no price, pos 1", _SP, price=None, position=1),
        _ranked("priced", _SP, price=99, position=7),
        _ranked("no price or position X", _SP),
        _ranked("no price or position Y", _SP),
    ]

    assert _ranked_titles(offers) == [
        "priced",
        "no price, pos 1",
        "no price or position X",
        "no price or position Y",
    ]


def test_related_products_come_after_all_exact_and_size_matches() -> None:
    offers = [
        _ranked("related 1", _REL, price=5, position=1),
        _ranked("exact", _SP, price=100, position=9),
        _ranked("related 2", _REL, price=6, position=2),
        _ranked("size", _SIZE, price=200, position=8),
    ]

    assert _ranked_titles(offers) == ["exact", "size", "related 1", "related 2"]


def test_ranking_excludes_different_variant() -> None:
    offers = [
        _ranked("exact", _SP, price=28),
        _ranked("saree", _DIFF, price=1),
        _ranked("titanium bar", _DIFF, price=2),
    ]

    assert _ranked_titles(offers) == ["exact"]


def test_duplicate_listing_from_same_retailer_keeps_cheapest() -> None:
    offers = [
        _ranked("Tata Salt 1 kg", _SP, price=35, position=1, source="amazon.in"),
        _ranked("TATA SALT, 1 KG", _SP, price=28, position=4, source="Amazon.in"),
    ]

    ranked = rank_comparison_offers(offers)

    assert len(ranked) == 1
    assert ranked[0].offer.extracted_price == 28


def test_different_sizes_and_configurations_from_one_retailer_are_kept() -> None:
    offers = [
        _ranked("Apple iPhone 17 Pro Max 256GB Black", _REL, price=149900, source="Croma"),
        _ranked("Apple iPhone 17 Pro Max 512GB Black", _REL, price=169900, source="Croma"),
        _ranked("Apple iPhone 17 Pro Max 256GB Silver", _REL, price=149900, source="Croma"),
    ]

    assert len(rank_comparison_offers(offers)) == 3


def test_same_title_from_different_retailers_is_not_a_duplicate() -> None:
    offers = [
        _ranked("Tata Salt 1 kg", _SP, price=28, source="Zepto"),
        _ranked("Tata Salt 1 kg", _SP, price=29, source="DMart Ready"),
    ]

    assert _ranked_titles(offers) == ["Tata Salt 1 kg", "Tata Salt 1 kg"]


def test_ranking_does_not_modify_its_input() -> None:
    offers = [_ranked("B", _SP, price=2), _ranked("A", _SP, price=1), _ranked("noise", _DIFF)]
    before = [c.offer.title for c in offers]

    rank_comparison_offers(offers)

    assert [c.offer.title for c in offers] == before


# --- retailer URL resolution before classification ------------------------------
# Each approved candidate's direct retailer URL is resolved (one
# google_immersive_product call) BEFORE Nemotron, so the retailer page can be
# read as evidence. That same URL is what the UI links to - nothing is resolved
# again afterwards. Unknown sellers are never resolved, and no failure in this
# step drops a candidate.

_GOOGLE_LINK = "https://www.google.com/shopping/somelisting"
_PAGE_TEXT = (
    "Product name: Tata Salt | Product description: Vacuum evaporated iodised salt, 1 kg pack"
)
_SEARCH = "shopping_comparison_stage.google_shopping_serpapi.search_shopping_offers"
_RESOLVE = "shopping_comparison_stage.google_shopping_serpapi.resolve_direct_merchant_link"
_FETCH_PAGE_TEXT = "shopping_comparison_stage.fetch_product_page_text"


def _fixed_classification_fn(classification: ShoppingMatchClassification) -> Any:
    def _classify(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        return [_verdict(classification) for _ in image_paths]

    return _classify


def _retailer_url(page_token: str, source: str | None) -> str:
    return f"https://retailer.example/{page_token}"


def _links(result: Any) -> dict[str | None, str]:
    stored = [c.offer for c in result.offers] + list(result.unclassified_offers)
    return {offer.page_token: str(offer.product_link) for offer in stored}


def test_approved_candidate_gets_retailer_url_before_nemotron() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offer = _offer(1, page_token="tok-1", source="Zepto", title="Tata Salt 1 kg")

    with (
        patch(_SEARCH, return_value=[offer]),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        result = _run(stage.compare("Tata Salt 1 kg"))

    mock_resolve.assert_called_once_with("tok-1", "Zepto")
    # The label Nemotron received already carries the retailer URL, so resolution came first.
    assert "URL: https://retailer.example/tok-1" in vision_model.classify_calls[0][0]
    assert str(result.offers[0].offer.product_link) == "https://retailer.example/tok-1"


@pytest.mark.parametrize("classification", list(ShoppingMatchClassification))
def test_resolved_url_is_kept_whatever_the_classification(
    classification: ShoppingMatchClassification,
) -> None:
    vision_model = _FakeVisionModel(classify_fn=_fixed_classification_fn(classification))
    stage = _stage(vision_model)

    with (
        patch(_SEARCH, return_value=[_offer(1, page_token="tok-1")]),
        patch(_RESOLVE, side_effect=_retailer_url),
    ):
        result = _run(
            stage.compare("Some Product 1 kg")
        )  # a size is stated, so "different size" stays as is

    assert result.offers[0].verdict.classification == classification
    assert str(result.offers[0].offer.product_link) == "https://retailer.example/tok-1"


def test_unknown_seller_never_gets_immersive_resolution() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [
        _offer(1, page_token="tok-unknown", source="icluster technologies"),
        _offer(2, page_token="tok-approved", source="Croma"),
        _offer(3, page_token="tok-shopsy", source="Shopsy By Flipkart"),
    ]

    with (
        patch(_SEARCH, return_value=offers),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        _run(stage.compare("Some Product"))

    mock_resolve.assert_called_once_with("tok-approved", "Croma")


def test_retailer_page_text_reaches_nemotron_with_the_original_query() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offer = _offer(1, page_token="tok-1", source="Zepto", title="Tata Salt 1 kg", price=28)

    with patch(_SEARCH, return_value=[offer]), patch(_RESOLVE, side_effect=_retailer_url):
        with patch(_FETCH_PAGE_TEXT, return_value=_PAGE_TEXT) as mock_fetch:
            _run(stage.compare("TATA SALT 1KG PKT"))

    # The retailer page (not Google's), checked against the candidate's own listing title.
    mock_fetch.assert_called_once_with("https://retailer.example/tok-1", about="Tata Salt 1 kg")
    # The original query, not the normalized search string (which drops "PKT").
    assert vision_model.product_names == ["TATA SALT 1KG PKT"]
    label = vision_model.classify_calls[0][0]
    for expected in (
        "Title: Tata Salt 1 kg",
        "Retailer: Zepto",
        "Price: Rs 28",
        "URL: https://retailer.example/tok-1",
        f"Page text: {_PAGE_TEXT}",
    ):
        assert expected in label
    assert vision_model.image_path_calls[0][0] is not None  # image still sent as a supplement


def test_page_extraction_failure_falls_back_and_still_classifies() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with (
        patch(_SEARCH, return_value=[_offer(1, page_token="tok-1")]),
        patch(_RESOLVE, side_effect=_retailer_url),
    ):
        with patch(_FETCH_PAGE_TEXT, return_value=None):
            result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 1
    assert result.unclassified_offers == []
    label = vision_model.classify_calls[0][0]
    assert "URL: https://retailer.example/tok-1" in label
    assert "Page text: not available" in label


@pytest.mark.parametrize("resolution", ["no matching store", "request raises"])
def test_immersive_resolution_failure_falls_back_and_still_classifies(resolution: str) -> None:
    async def raising_resolve(*args: Any, **kwargs: Any) -> str | None:
        raise RuntimeError("simulated immersive-product API failure")

    if resolution == "no matching store":
        resolve_patch = patch(_RESOLVE, return_value=None)
    else:
        resolve_patch = patch(_RESOLVE, side_effect=raising_resolve)
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with patch(_SEARCH, return_value=[_offer(1, page_token="tok-1")]), resolve_patch:
        # The real page-text function: an unresolved (Google) link must not even be fetched.
        with patch(_FETCH_PAGE_TEXT, new=product_page_text.fetch_product_page_text):
            with patch("product_page_text._fetch_html") as mock_fetch_html:
                result = _run(stage.compare("Some Product"))

    mock_fetch_html.assert_not_called()
    assert len(result.offers) == 1  # not dropped
    assert str(result.offers[0].offer.product_link) == _GOOGLE_LINK
    label = vision_model.classify_calls[0][0]
    assert "URL: not available" in label  # Google's link is never shown to the model
    assert "Page text: not available" in label
    assert "Title: Offer 1" in label


def test_candidate_without_a_page_token_is_not_resolved_but_still_classified() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with patch(_SEARCH, return_value=[_offer(1, page_token=None)]), patch(_RESOLVE) as mock_resolve:
        result = _run(stage.compare("Some Product"))

    mock_resolve.assert_not_called()
    assert len(result.offers) == 1
    assert str(result.offers[0].offer.product_link) == _GOOGLE_LINK


def test_resolved_url_is_reused_after_classification_without_a_second_call() -> None:
    vision_model = _FakeVisionModel()  # everything same_product, i.e. all visible
    stage = _stage(vision_model)
    offers = [_offer(i, page_token=f"tok-{i}", source="Zepto") for i in range(3)]

    with (
        patch(_SEARCH, return_value=offers),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        result = _run(stage.compare("Some Product"))

    assert (
        mock_resolve.call_count == 3
    )  # once per candidate, before classification - never again after
    visible = rank_comparison_offers(result.offers)
    assert sorted(str(c.offer.product_link) for c in visible) == [
        f"https://retailer.example/tok-{i}" for i in range(3)
    ]


def test_candidates_sharing_a_token_and_seller_share_one_immersive_call() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [
        _offer(1, page_token="tok-shared", source="Zepto", title="Tata Salt 1 kg"),
        _offer(2, page_token="tok-shared", source="zepto", title="Tata Salt 1 kg pouch"),
        _offer(3, page_token="tok-other", source="Zepto", title="Tata Salt 2 kg"),
    ]

    with (
        patch(_SEARCH, return_value=offers),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        result = _run(stage.compare("Tata Salt 1 kg"))

    assert sorted(call.args[0] for call in mock_resolve.call_args_list) == [
        "tok-other",
        "tok-shared",
    ]
    links = [str(c.offer.product_link) for c in result.offers]
    assert links.count("https://retailer.example/tok-shared") == 2


def test_image_remains_optional_with_retailer_page_text() -> None:
    vision_model = _FakeVisionModel()
    offer = _offer(1, page_token="tok-1", url="https://example.com/bad.jpg")
    stage = _stage(
        vision_model, downloader=_FakeDownloader(fail_urls={"https://example.com/bad.jpg"})
    )

    with patch(_SEARCH, return_value=[offer]), patch(_RESOLVE, side_effect=_retailer_url):
        with patch(_FETCH_PAGE_TEXT, return_value=_PAGE_TEXT):
            result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 1
    assert vision_model.image_path_calls[0] == [None]
    assert f"Page text: {_PAGE_TEXT}" in vision_model.classify_calls[0][0]


def test_candidate_with_no_url_page_text_or_image_is_still_classified() -> None:
    vision_model = _FakeVisionModel()
    offer = _offer(1, page_token=None, url="https://example.com/bad.jpg")
    stage = _stage(
        vision_model, downloader=_FakeDownloader(fail_urls={"https://example.com/bad.jpg"})
    )

    with patch(_SEARCH, return_value=[offer]):
        result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 1
    assert vision_model.image_path_calls[0] == [None]


def test_one_shopping_search_and_one_immersive_call_per_approved_candidate() -> None:
    """End to end through the real resolver, counting actual SerpApi HTTP requests."""
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = [_offer(i, page_token=f"tok-{i}", source="Zepto") for i in range(3)] + [
        _offer(9, page_token="tok-unknown", source="Local Store")
    ]
    stores_payload = {
        "product_results": {
            "stores": [{"name": "Zepto", "link": "https://www.zepto.example/pn/item"}]
        }
    }

    with patch(_SEARCH, return_value=offers) as mock_search, patch(_RESOLVE, new=_REAL_RESOLVE):
        with patch(
            "search.retailers.google_shopping_serpapi._fetch_json", return_value=stores_payload
        ) as mock_http:
            with patch.dict("os.environ", {"PIR_SERPAPI_API_KEY": "test-key"}):
                result = _run(stage.compare("Some Product"))

    mock_search.assert_called_once()  # no second Google Shopping search
    requested = [call.args[0] for call in mock_http.call_args_list]
    assert (
        len(requested) == 3
    )  # one per approved candidate, none for the unknown seller, none repeated
    assert all("engine=google_immersive_product" in url for url in requested)
    assert not any("tok-unknown" in url for url in requested)
    assert all(
        str(c.offer.product_link) == "https://www.zepto.example/pn/item" for c in result.offers
    )


# --- ranking and stored results with the new order -----------------------------


def _classify_by_title(
    image_paths: list[Path | None], offer_labels: list[str]
) -> list[ShoppingMatchVerdict]:
    if any("jar" in label for label in offer_labels):
        raise RuntimeError("simulated NVIDIA Build failure for this batch")

    def classification(label: str) -> ShoppingMatchClassification:
        if "Saree" in label:
            return _DIFF
        if "Lite" in label:
            return _REL
        if "2 kg" in label:
            return _SIZE
        return _SP

    return [_verdict(classification(label)) for label in offer_labels]


def _mixed_offers() -> list[ShoppingOffer]:
    return [
        _offer(1, page_token="tok-dup-expensive", source="Zepto", title="Tata Salt 1 kg", price=30),
        _offer(2, page_token="tok-exact", source="zepto", title="TATA SALT, 1 KG", price=28),
        _offer(3, page_token="tok-size", source="Zepto", title="Tata Salt 2 kg", price=55),
        _offer(4, page_token="tok-related", source="Zepto", title="Tata Lite 1 kg", price=39),
        _offer(5, page_token="tok-irrelevant", source="Zepto", title="Silk Saree", price=5),
        # Sixth offer = its own batch of one, which `_classify_by_title` fails -> unverified.
        _offer(6, page_token="tok-unverified", source="Zepto", title="Tata Salt 1 kg jar"),
    ]


def test_visible_order_is_unchanged_and_every_stored_offer_keeps_its_resolved_url() -> None:
    vision_model = _FakeVisionModel(classify_fn=_classify_by_title)
    stage = _stage(vision_model)

    with (
        patch(_SEARCH, return_value=_mixed_offers()),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        result = _run(stage.compare("Tata Salt 1 kg"))

    assert mock_resolve.call_count == 6  # every approved candidate, once
    visible = rank_comparison_offers(result.offers)
    assert [(c.offer.title, c.offer.extracted_price) for c in visible] == [
        ("TATA SALT, 1 KG", 28),
        ("Tata Salt 2 kg", 55),
        ("Tata Lite 1 kg", 39),
    ]
    assert [str(c.offer.product_link) for c in visible] == [
        "https://retailer.example/tok-exact",
        "https://retailer.example/tok-size",
        "https://retailer.example/tok-related",
    ]
    # Everything is still stored for debugging, each with its own resolved URL.
    assert len(result.offers) == 5
    assert [o.title for o in result.unclassified_offers] == ["Tata Salt 1 kg jar"]
    assert _links(result) == {
        token: f"https://retailer.example/{token}" for token in _links(result)
    }


def test_classification_failure_after_resolution_keeps_the_offer_as_unverified() -> None:
    def failing_classify(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        raise RuntimeError("simulated NVIDIA Build failure")

    vision_model = _FakeVisionModel(classify_fn=failing_classify)
    stage = _stage(vision_model)

    with (
        patch(_SEARCH, return_value=[_offer(1, page_token="tok-1")]),
        patch(_RESOLVE, side_effect=_retailer_url),
    ):
        result = _run(stage.compare("Some Product"))

    assert result.offers == []
    assert len(result.unclassified_offers) == 1
    assert str(result.unclassified_offers[0].product_link) == "https://retailer.example/tok-1"


# --- candidate cap ------------------------------------------------------------
# Only the first N approved candidates (Google's order) go through the
# expensive part of the pipeline: an immersive call, a retailer-page fetch
# and Nemotron. Approved candidates beyond the cap are not processed or shown.


def _approved_offers(count: int) -> list[ShoppingOffer]:
    return [
        _offer(i, page_token=f"tok-{i}", url=f"https://example.com/{i}.jpg") for i in range(count)
    ]


def test_default_cap_is_ten() -> None:
    assert shopping_comparison_stage._MAX_CANDIDATES == 10


def test_at_most_ten_approved_candidates_are_resolved_in_google_order() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with (
        patch(_SEARCH, return_value=_approved_offers(15)),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        result = _run(stage.compare("Some Product"))

    assert [call.args[0] for call in mock_resolve.call_args_list] == [f"tok-{i}" for i in range(10)]
    assert _links(result) == {f"tok-{i}": f"https://retailer.example/tok-{i}" for i in range(10)}


def test_candidates_beyond_the_cap_are_not_resolved_fetched_or_classified() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with (
        patch(_SEARCH, return_value=_approved_offers(15)),
        patch(_RESOLVE, side_effect=_retailer_url),
    ):
        with patch(_FETCH_PAGE_TEXT, new=product_page_text.fetch_product_page_text):
            with patch(
                "product_page_text._fetch_html", return_value="<title>Product page</title>"
            ) as mock_html:
                result = _run(stage.compare("Some Product"))

    fetched = sorted(call.args[0] for call in mock_html.call_args_list)
    assert fetched == sorted(f"https://retailer.example/tok-{i}" for i in range(10))
    labels = [label for batch in vision_model.classify_calls for label in batch]
    assert len(labels) == 10
    assert not any("Title: Offer 12" in label for label in labels)
    stored_titles = {c.offer.title for c in result.offers} | {
        o.title for o in result.unclassified_offers
    }
    assert stored_titles == {f"Offer {i}" for i in range(10)}


def test_unknown_sellers_do_not_use_up_the_cap() -> None:
    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)
    offers = []
    for i in range(10):
        offers.append(_offer(100 + i, page_token=f"tok-unknown-{i}", source=f"Local Store {i}"))
        offers.append(_offer(i, page_token=f"tok-{i}"))

    with (
        patch(_SEARCH, return_value=offers),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        result = _run(stage.compare("Some Product"))

    assert [call.args[0] for call in mock_resolve.call_args_list] == [f"tok-{i}" for i in range(10)]
    assert len(result.offers) == 10


def test_cap_is_configurable() -> None:
    vision_model = _FakeVisionModel()
    stage = ShoppingComparisonStage(
        downloader=_FakeDownloader(),
        vision_model=vision_model,  # type: ignore[arg-type]
        storage=_FakeStorage(),  # type: ignore[arg-type]
        max_candidates=3,
    )

    with (
        patch(_SEARCH, return_value=_approved_offers(8)),
        patch(_RESOLVE, side_effect=_retailer_url) as mock_resolve,
    ):
        result = _run(stage.compare("Some Product"))

    assert [call.args[0] for call in mock_resolve.call_args_list] == ["tok-0", "tok-1", "tok-2"]
    assert len(result.offers) == 3


def test_fallbacks_still_apply_within_the_cap() -> None:
    """A failed resolution inside the cap keeps Google's link and the candidate is still classified."""

    def resolve_only_even(page_token: str, source: str | None) -> str | None:
        return _retailer_url(page_token, source) if int(page_token.split("-")[1]) % 2 == 0 else None

    vision_model = _FakeVisionModel()
    stage = _stage(vision_model)

    with (
        patch(_SEARCH, return_value=_approved_offers(4)),
        patch(_RESOLVE, side_effect=resolve_only_even),
    ):
        result = _run(stage.compare("Some Product"))

    assert len(result.offers) == 4
    links = _links(result)
    assert links["tok-0"] == "https://retailer.example/tok-0"
    assert links["tok-1"] == _GOOGLE_LINK


# --- verdicts must agree with the model's own attribute answers -------------------
# Live run: an empty ketchup bottle was labelled "related" to a ketchup query
# while the model answered brand_match=False and product_match=False.


def _flagged_verdict(
    classification: ShoppingMatchClassification, brand: bool | None, product: bool | None
) -> ShoppingMatchVerdict:
    return ShoppingMatchVerdict(
        classification=classification,
        confidence=0.9,
        reason="fake",
        brand_match=brand,
        product_match=product,
        model_name="fake",
    )


@pytest.mark.parametrize("classification", [_SP, _SIZE, _REL])
def test_relevant_verdict_with_neither_brand_nor_product_match_becomes_not_relevant(
    classification: ShoppingMatchClassification,
) -> None:
    reconciled = shopping_comparison_stage._reconcile_verdict(
        _flagged_verdict(classification, False, False), _offer(1), "Some Product 1 kg"
    )

    assert reconciled.classification == _DIFF
    assert reconciled.reason == "fake"  # the model's explanation is kept


@pytest.mark.parametrize(
    ("classification", "brand", "product"),
    [
        (_REL, False, True),  # cross-brand alternative, e.g. another brand's muesli
        (_REL, True, False),
        (_SP, True, True),
        (_REL, None, None),  # answers missing (old-shape or recovered response)
        (_REL, False, None),
        (_DIFF, False, False),
    ],
)
def test_consistent_or_incomplete_verdicts_are_left_untouched(
    classification: ShoppingMatchClassification, brand: bool | None, product: bool | None
) -> None:
    verdict = _flagged_verdict(classification, brand, product)

    assert (
        shopping_comparison_stage._reconcile_verdict(verdict, _offer(1), "Some Product 1 kg")
        is verdict
    )


def test_self_contradicting_related_verdict_is_hidden_end_to_end() -> None:
    def classify_fn(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        return [_flagged_verdict(_REL, False, False), _flagged_verdict(_REL, False, True)]

    vision_model = _FakeVisionModel(classify_fn=classify_fn)
    stage = _stage(vision_model)
    offers = [
        _offer(1, title="Empty Ketchup Bottle"),
        _offer(2, title="Other Brand Tomato Ketchup"),
    ]

    with patch(_SEARCH, return_value=offers):
        result = _run(stage.compare("Kissan Fresh Tomato Ketchup"))

    assert [c.offer.title for c in rank_comparison_offers(result.offers)] == [
        "Other Brand Tomato Ketchup"
    ]
    assert len(result.offers) == 2  # the hidden one is still stored


# --- container / accessory listings are not the product ------------------------------
# Live runs: for "Kissan Fresh Tomato Ketchup", with no Kissan product in the
# pool, the model called an EMPTY ketchup bottle "related" and then the "same
# product" (claiming every attribute matched). The title itself says what it is.

_VELURA_TITLE = (
    "VELURA Mini Tomato Ketchup Bottle Containers Bottle for Kitchen BBQ Outdoor ketchup|"
    "1 Empty Bottle(AS YOU choose)| Other Kitchen Tools & Gadgets|Home"
)


@pytest.mark.parametrize(
    "title",
    [
        _VELURA_TITLE,
        "Empty Squeeze Sauce Bottle 250ml",
        "Glass Jar Only - Replacement",
        "Ketchup Dispenser Cap only",
        "Empty packaging box for gifting",
        "Spigen Ultra Hybrid Case for iPhone 17",
        "Tempered Glass Screen Protector for iPhone 17",
        "iPhone 17 Accessories Kit",
    ],
)
def test_container_and_accessory_listings_are_detected(title: str) -> None:
    assert shopping_comparison_stage._is_container_or_accessory_listing(
        title, "Kissan Fresh Tomato Ketchup"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Kissan Fresh Tomato Ketchup Bottle 950g",
        "Kissan Fresh Tomato Ketchup Doy Pack 1 kg",
        "Tata Salt Pet Jar Free flowing & Iodised Namak",
        "MAGGI 2-Minute Instant Noodles 1.26kg Pouch (Pack of 18, 70 g each)",
        "Colgate Strong Teeth Toothpaste 200g (Packaging may vary)",
        "Dabur Honey 500g Squeezy Pack with Flip Cap",
        "Coca-Cola Can 300 ml",
        "Apple iPhone 17 256GB Black",
    ],
)
def test_ordinary_packaged_products_are_not_treated_as_containers(title: str) -> None:
    assert not shopping_comparison_stage._is_container_or_accessory_listing(title, "Some Product")


@pytest.mark.parametrize(
    "query", ["empty ketchup bottle", "case for iPhone 17", "iPhone 17 accessories"]
)
def test_container_rule_is_off_when_the_query_asks_for_a_container_or_accessory(query: str) -> None:
    assert not shopping_comparison_stage._is_container_or_accessory_listing(_VELURA_TITLE, query)
    assert not shopping_comparison_stage._is_container_or_accessory_listing(
        "Spigen Case for iPhone 17", query
    )


@pytest.mark.parametrize("classification", [_SP, _SIZE, _REL])
def test_relevant_verdict_on_a_container_listing_becomes_not_relevant(
    classification: ShoppingMatchClassification,
) -> None:
    """Even when the model claims every attribute matches, as it did live."""
    verdict = _flagged_verdict(classification, True, True)
    offer = _offer(1, title=_VELURA_TITLE)

    reconciled = shopping_comparison_stage._reconcile_verdict(
        verdict, offer, "Kissan Fresh Tomato Ketchup"
    )

    assert reconciled.classification == _DIFF
    assert reconciled.reason == "fake"


def test_no_genuine_product_in_the_pool_leaves_nothing_to_show() -> None:
    """The Kissan case end to end: only an accessory is matched by the model, so nothing is visible."""

    def classify_fn(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        return [
            _flagged_verdict(_SP, True, True)
            if "Empty Bottle" in label
            else _flagged_verdict(_DIFF, False, False)
            for label in offer_labels
        ]

    vision_model = _FakeVisionModel(classify_fn=classify_fn)
    stage = _stage(vision_model)
    offers = [
        _offer(1, title="Red by Kiss Professional Edge Brush"),
        _offer(2, title=_VELURA_TITLE),
        _offer(3, title="Haus & Kinder Hip Seat Baby Carrier"),
    ]

    with patch(_SEARCH, return_value=offers):
        result = _run(stage.compare("Kissan Fresh Tomato Ketchup"))

    assert rank_comparison_offers(result.offers) == []
    assert all(c.verdict.classification == _DIFF for c in result.offers)
    assert len(result.offers) == 3  # still stored
    assert result.search_failed is False


# --- size semantics ------------------------------------------------------------------
# With no size in the query, a candidate cannot be "the same product in a
# different size" - it is simply the same product.


@pytest.mark.parametrize(
    ("query", "states_size"),
    [
        ("Tata Salt", False),
        ("Maggi", False),
        ("iPhone 17", False),
        ("Maggi 2 Minute Noodles", False),
        ("Bournvita 5 Star Magic", False),
        ("Tata Salt 1 kg", True),
        ("Kissan Fresh Tomato Ketchup 950g", True),
        ("Amul Milk 500 ml", True),
        ("iPhone 17 256GB", True),
        ("Coca-Cola pack of 6", True),
        ("Maggi Noodles 12 pcs", True),
        ("Maggi 6 x 70g", True),
    ],
)
def test_query_states_size(query: str, states_size: bool) -> None:
    assert shopping_comparison_stage._query_states_size(query) is states_size


def test_different_size_verdict_becomes_same_product_when_query_has_no_size() -> None:
    verdict = _flagged_verdict(_SIZE, True, True)

    reconciled = shopping_comparison_stage._reconcile_verdict(
        verdict, _offer(1, title="Maggi Noodles 560 g"), "Maggi"
    )

    assert reconciled.classification == _SP


def test_different_size_verdict_is_kept_when_query_states_a_size() -> None:
    verdict = _flagged_verdict(_SIZE, True, True)

    reconciled = shopping_comparison_stage._reconcile_verdict(
        verdict, _offer(1, title="Tata Salt 2 kg"), "Tata Salt 1 kg"
    )

    assert reconciled is verdict


def test_sizeless_query_shows_every_size_as_same_product_end_to_end() -> None:
    def classify_fn(
        image_paths: list[Path | None], offer_labels: list[str]
    ) -> list[ShoppingMatchVerdict]:
        return [
            _flagged_verdict(_SIZE if "1.26kg" in label else _SP, True, True)
            for label in offer_labels
        ]

    vision_model = _FakeVisionModel(classify_fn=classify_fn)
    stage = _stage(vision_model)
    offers = [
        _offer(1, title="Maggi Masala Noodles 1.26kg Pouch", price=226),
        _offer(2, title="Maggi Masala Noodles 70 g", price=15),
    ]

    with patch(_SEARCH, return_value=offers):
        result = _run(stage.compare("Maggi"))

    visible = rank_comparison_offers(result.offers)
    assert [c.verdict.classification for c in visible] == [_SP, _SP]
    assert [c.offer.extracted_price for c in visible] == [15, 226]  # one group, cheapest first
