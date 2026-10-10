"""
Unit tests for `search/retailers/google_shopping_serpapi.py` - the SerpApi-
based Google Shopping retailer module (second-priority active retailer,
after Amazon, before the Google Images fallback).

All SerpApi HTTP calls are mocked by patching `google_shopping_serpapi._fetch_json`
(the module's one blocking-I/O boundary) or by unit-testing its error-
classification helpers directly - no live network calls, no real API key
required. Mirrors `tests/test_amazon_serpapi.py`'s structure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from unittest.mock import patch

import pytest

from search.retailers import google_shopping_serpapi
from search.serpapi import SerpApiInvalidApiKeyError, SerpApiRateLimitError, SerpApiSearchError

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


def _set_api_key(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    # See tests/test_amazon_serpapi.py's `_set_api_key` for why this must be
    # an explicit empty string, not env-var deletion, when a real `.env`
    # with a real key exists on disk.
    monkeypatch.setenv("PIR_SERPAPI_API_KEY", value or "")


# --- search_product_page ----------------------------------------------------


def test_search_product_page_builds_detail_urls_from_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    shopping_results = [
        {"position": i, "title": f"Product {i}", "immersive_product_page_token": f"token{i}"}
        for i in range(1, 8)
    ]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": shopping_results}
    ) as mock_fetch:
        result = _run(google_shopping_serpapi.search_product_page(None, "test query"))

    assert len(result) == 5  # capped at _MAX_CANDIDATES
    for url, i in zip(result, range(1, 6)):
        assert f"page_token=token{i}" in url
        assert "engine=google_immersive_product" in url
        assert "api_key" not in url  # never leak the key into a persisted/displayed URL

    called_url = mock_fetch.call_args[0][0]
    assert "engine=google_shopping" in called_url
    assert "gl=in" in called_url


def test_search_product_page_skips_candidates_without_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    shopping_results = [
        {"title": "No token here"},
        {"title": "Has a token", "immersive_product_page_token": "abc123"},
    ]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": shopping_results}
    ):
        result = _run(google_shopping_serpapi.search_product_page(None, "q"))

    assert len(result) == 1
    assert "page_token=abc123" in result[0]


def test_search_product_page_returns_empty_list_when_no_shopping_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value={}):
        result = _run(google_shopping_serpapi.search_product_page(None, "q"))
    assert result == []


def test_search_product_page_dedupes_identical_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    shopping_results = [
        {"immersive_product_page_token": "same-token"},
        {"immersive_product_page_token": "same-token"},
    ]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": shopping_results}
    ):
        result = _run(google_shopping_serpapi.search_product_page(None, "q"))
    assert len(result) == 1


# --- extract_gallery ---------------------------------------------------------


def test_extract_gallery_reads_all_thumbnails(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    thumbnails = [f"https://encrypted-tbn{i}.gstatic.com/shopping?q=tbn:img{i}" for i in range(10)]
    with patch.object(
        google_shopping_serpapi,
        "_fetch_json",
        return_value={"product_results": {"title": "Some Product", "thumbnails": thumbnails}},
    ) as mock_fetch:
        product_url = google_shopping_serpapi._build_detail_url("my-token")
        gallery = _run(google_shopping_serpapi.extract_gallery(None, product_url))

    assert gallery.retailer == "Google Shopping"
    assert gallery.image_urls == thumbnails
    assert gallery.product_title == "Some Product"
    assert "thumbnails" in gallery.source_description
    called_url = mock_fetch.call_args[0][0]
    assert "engine=google_immersive_product" in called_url
    assert "page_token=my-token" in called_url


def test_extract_gallery_raises_on_url_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    with pytest.raises(SerpApiSearchError):
        _run(
            google_shopping_serpapi.extract_gallery(
                None, "https://serpapi.com/search.json?engine=google_immersive_product"
            )
        )


def test_extract_gallery_empty_thumbnails_returns_empty_gallery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value={"product_results": {}}):
        product_url = google_shopping_serpapi._build_detail_url("my-token")
        gallery = _run(google_shopping_serpapi.extract_gallery(None, product_url))
    assert gallery.image_urls == []
    assert gallery.product_title is None


# --- API key handling ---------------------------------------------------------


def test_missing_api_key_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, None)
    with pytest.raises(SerpApiInvalidApiKeyError):
        _run(google_shopping_serpapi.search_product_page(None, "q"))


# --- error classification (direct unit tests of the translation helpers) -----


def test_raise_for_http_error_401_is_invalid_api_key() -> None:
    with pytest.raises(SerpApiInvalidApiKeyError):
        google_shopping_serpapi._raise_for_http_error(401, b'{"error": "Invalid API key"}')


def test_raise_for_http_error_429_is_rate_limit() -> None:
    with pytest.raises(SerpApiRateLimitError):
        google_shopping_serpapi._raise_for_http_error(429, b'{"error": "Rate limit exceeded"}')


def test_raise_for_error_payload_dead_engine_is_generic_error() -> None:
    # Confirmed live: engine=google_product returns exactly this message.
    with pytest.raises(SerpApiSearchError):
        google_shopping_serpapi._raise_for_error_payload(
            {"error": "The Google Product service is no longer offered by Google."}
        )


def test_raise_for_error_payload_no_error_field_is_a_noop() -> None:
    google_shopping_serpapi._raise_for_error_payload({"shopping_results": []})  # must not raise


# --- page_token round-trip ----------------------------------------------------


def test_build_and_extract_page_token_round_trip() -> None:
    url = google_shopping_serpapi._build_detail_url("abc-123-XYZ")
    assert google_shopping_serpapi._extract_page_token(url) == "abc-123-XYZ"
    assert "api_key" not in url


# --- search_shopping_offers (cheap comparison-list path, no detail engine) ---


def _shopping_result(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "position": 1,
        "title": "Tata Salt 1 kg",
        "source": "Zepto",
        "price": "₹28",
        "extracted_price": 28.0,
        "thumbnail": "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:abc",
        "product_link": "https://www.google.co.in/search?ibp=oshop&q=tata+salt",
        "immersive_product_page_token": "token-abc-123",
    }
    base.update(overrides)
    return base


def test_search_shopping_offers_reads_fields_directly_no_detail_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    results = [_shopping_result()]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": results}
    ) as mock_fetch:
        offers = _run(google_shopping_serpapi.search_shopping_offers("tata salt", max_results=10))

    assert len(offers) == 1
    offer = offers[0]
    assert offer.title == "Tata Salt 1 kg"
    assert offer.source == "Zepto"
    assert offer.price == "₹28"
    assert offer.extracted_price == 28.0
    assert str(offer.thumbnail_url) == "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:abc"
    # Only ONE call - engine=google_shopping - never the detail engine.
    assert mock_fetch.call_count == 1
    assert "engine=google_shopping" in mock_fetch.call_args[0][0]


def test_search_shopping_offers_caps_at_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    results = [_shopping_result(position=i) for i in range(20)]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": results}
    ):
        offers = _run(google_shopping_serpapi.search_shopping_offers("q", max_results=5))
    assert len(offers) == 5


def test_search_shopping_offers_skips_entries_missing_title_or_thumbnail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    results = [
        _shopping_result(title=None),
        _shopping_result(thumbnail=None),
        _shopping_result(),  # complete
    ]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": results}
    ):
        offers = _run(google_shopping_serpapi.search_shopping_offers("q", max_results=10))
    assert len(offers) == 1


def test_search_shopping_offers_skips_malformed_thumbnail_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    results = [_shopping_result(thumbnail="not-a-url"), _shopping_result()]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": results}
    ):
        offers = _run(google_shopping_serpapi.search_shopping_offers("q", max_results=10))
    assert len(offers) == 1


def test_search_shopping_offers_returns_empty_list_when_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value={}):
        offers = _run(google_shopping_serpapi.search_shopping_offers("q", max_results=10))
    assert offers == []


def test_search_shopping_offers_captures_page_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    results = [_shopping_result(immersive_product_page_token="my-token-xyz")]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": results}
    ):
        offers = _run(google_shopping_serpapi.search_shopping_offers("q", max_results=10))
    assert offers[0].page_token == "my-token-xyz"


def test_search_shopping_offers_page_token_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    results = [_shopping_result(immersive_product_page_token=None)]
    with patch.object(
        google_shopping_serpapi, "_fetch_json", return_value={"shopping_results": results}
    ):
        offers = _run(google_shopping_serpapi.search_shopping_offers("q", max_results=10))
    assert offers[0].page_token is None


# --- resolve_direct_merchant_link --------------------------------------------


def test_resolve_direct_merchant_link_finds_matching_store(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    payload = {
        "product_results": {
            "stores": [
                {"name": "amazon.in", "link": "https://www.amazon.in/dp/B000000000"},
                {"name": "Zepto", "link": "https://www.zepto.com/pn/some-product"},
            ]
        }
    }
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value=payload) as mock_fetch:
        link = _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", "Zepto"))
    assert link == "https://www.zepto.com/pn/some-product"
    assert "engine=google_immersive_product" in mock_fetch.call_args[0][0]
    assert "page_token=token-1" in mock_fetch.call_args[0][0]


@pytest.mark.parametrize(
    ("source", "store_name"),
    [
        ("amazon.in", "Amazon.in"),
        ("Amazon.in", "Amazon"),
        ("Amazon", "Amazon India"),
        ("JioMart Grocery", "JioMart"),
        ("DMart Ready", "dmart.in"),
        ("Apple", "Apple Store"),
    ],
)
def test_resolve_direct_merchant_link_matches_the_same_canonical_retailer(
    monkeypatch: pytest.MonkeyPatch, source: str, store_name: str
) -> None:
    _set_api_key(monkeypatch, "test-key")
    payload = {
        "product_results": {
            "stores": [{"name": store_name, "link": "https://retailer.example/p/1"}]
        }
    }
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value=payload):
        link = _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", source))
    assert link == "https://retailer.example/p/1"


@pytest.mark.parametrize(
    ("source", "store_name"),
    [
        ("Apple", "Imagine Apple Premium Reseller"),
        ("Apple", "Apple Premium Reseller - Unicorn"),
        ("Flipkart", "Shopsy By Flipkart"),
        ("Tata CLiQ", "Tata CLiQ Fashion"),
        ("Amazon.in", "Amazonia Organics"),
        ("Samsung", "Samsung Authorized Reseller"),
    ],
)
def test_resolve_direct_merchant_link_rejects_lookalike_store_names(
    monkeypatch: pytest.MonkeyPatch, source: str, store_name: str
) -> None:
    """No substring matching: a store that merely contains the retailer's name is a different seller."""
    _set_api_key(monkeypatch, "test-key")
    payload = {
        "product_results": {
            "stores": [{"name": store_name, "link": "https://reseller.example/p/1"}]
        }
    }
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value=payload):
        link = _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", source))
    assert link is None


def test_resolve_direct_merchant_link_skips_a_reseller_listed_before_the_real_retailer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    payload = {
        "product_results": {
            "stores": [
                {
                    "name": "Imagine Apple Premium Reseller",
                    "link": "https://reseller.example/iphone",
                },
                {"name": "Apple", "link": "https://www.apple.com/in/shop/buy-iphone"},
            ]
        }
    }
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value=payload):
        link = _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", "Apple"))
    assert link == "https://www.apple.com/in/shop/buy-iphone"


def test_resolve_direct_merchant_link_makes_no_request_for_a_non_approved_seller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(google_shopping_serpapi, "_fetch_json") as mock_fetch:
        link = _run(
            google_shopping_serpapi.resolve_direct_merchant_link("token-1", "icluster technologies")
        )
    assert link is None
    mock_fetch.assert_not_called()


def test_resolve_direct_merchant_link_returns_none_when_source_not_in_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    payload = {
        "product_results": {
            "stores": [{"name": "Flipkart", "link": "https://www.flipkart.com/p/x"}]
        }
    }
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value=payload):
        link = _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", "Zepto"))
    assert link is None


def test_resolve_direct_merchant_link_returns_none_when_stores_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(google_shopping_serpapi, "_fetch_json", return_value={"product_results": {}}):
        link = _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", "Zepto"))
    assert link is None


def test_resolve_direct_merchant_link_returns_none_on_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(
        google_shopping_serpapi, "_fetch_json", side_effect=RuntimeError("network error")
    ):
        link = _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", "Zepto"))
    assert link is None


def test_resolve_direct_merchant_link_returns_none_for_missing_token_or_source() -> None:
    assert _run(google_shopping_serpapi.resolve_direct_merchant_link("", "Zepto")) is None
    assert _run(google_shopping_serpapi.resolve_direct_merchant_link("token-1", None)) is None
