"""
Unit tests for `search/retailers/amazon_serpapi.py` - the SerpApi-based
Amazon search/gallery module.

All SerpApi HTTP calls are mocked by patching `amazon_serpapi._fetch_json`
(the module's one blocking-I/O boundary) or by unit-testing its error-
classification helpers directly - no live network calls, no real API key
required. See HANDOFF.md / the implementation plan for the separate LIVE
verification pass against real SerpApi endpoints, which is what actually
establishes whether `product_results.thumbnails[]` is a complete gallery -
that is not something a mocked test can prove either way.

Async module functions are invoked via `asyncio.run(...)` inside plain
`def test_...()` functions rather than `@pytest.mark.asyncio`, since
pytest-asyncio isn't actually installed in this environment despite being a
declared dev dependency in `pyproject.toml` - this avoids making these tests
depend on that.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from unittest.mock import patch

import pytest

from search.retailers import amazon_serpapi
from search.serpapi import SerpApiInvalidApiKeyError, SerpApiRateLimitError, SerpApiSearchError

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


def _set_api_key(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    # `Settings` reads `.env` directly (pydantic-settings' `env_file`
    # source), so merely deleting the OS env var doesn't simulate "no key"
    # when a real `.env` with a real key exists on disk - an explicitly-set
    # empty string DOES take precedence over `.env`, so that's what "no key"
    # means here, not deletion.
    monkeypatch.setenv("PIR_SERPAPI_API_KEY", value or "")


# --- search_product_page ----------------------------------------------------


def test_search_product_page_filters_sponsored_and_caps_at_five(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    organic_results = [
        {"asin": "SPON", "link": "https://www.amazon.in/dp/SPON123456", "sponsored": True},
        *[
            {
                "asin": f"A{i}23456789"[:10],
                "link_clean": f"https://www.amazon.in/dp/A{i}23456789",
                "sponsored": False,
            }
            for i in range(7)
        ],
    ]
    with patch.object(
        amazon_serpapi, "_fetch_json", return_value={"organic_results": organic_results}
    ) as mock_fetch:
        result = _run(amazon_serpapi.search_product_page(None, "test query"))  # type: ignore[arg-type]

    assert result == [f"https://www.amazon.in/dp/A{i}23456789" for i in range(5)]
    called_url = mock_fetch.call_args[0][0]
    assert "engine=amazon" in called_url
    assert "amazon_domain=amazon.in" in called_url
    assert "k=test" in called_url


def test_search_product_page_prefers_link_clean_over_link(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    organic_results = [
        {
            "asin": "A123456789",
            "link": "https://www.amazon.in/dp/A123456789?ref=tracking",
            "link_clean": "https://www.amazon.in/dp/A123456789",
            "sponsored": False,
        }
    ]
    with patch.object(amazon_serpapi, "_fetch_json", return_value={"organic_results": organic_results}):
        result = _run(amazon_serpapi.search_product_page(None, "q"))  # type: ignore[arg-type]
    assert result == ["https://www.amazon.in/dp/A123456789"]


def test_search_product_page_returns_empty_list_when_no_organic_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(amazon_serpapi, "_fetch_json", return_value={}):
        result = _run(amazon_serpapi.search_product_page(None, "q"))  # type: ignore[arg-type]
    assert result == []


def test_search_product_page_dedupes_identical_links(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    organic_results = [
        {"asin": "A123456789", "link": "https://www.amazon.in/dp/A123456789", "sponsored": False},
        {"asin": "A123456789", "link": "https://www.amazon.in/dp/A123456789", "sponsored": False},
    ]
    with patch.object(amazon_serpapi, "_fetch_json", return_value={"organic_results": organic_results}):
        result = _run(amazon_serpapi.search_product_page(None, "q"))  # type: ignore[arg-type]
    assert result == ["https://www.amazon.in/dp/A123456789"]


# --- extract_gallery ---------------------------------------------------------


def test_extract_gallery_reads_all_thumbnails(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    thumbnails = [f"https://m.media-amazon.com/images/I/img{i}.jpg" for i in range(7)]
    with patch.object(
        amazon_serpapi, "_fetch_json", return_value={"product_results": {"thumbnails": thumbnails}}
    ) as mock_fetch:
        gallery = _run(
            amazon_serpapi.extract_gallery(None, "https://www.amazon.in/dp/B0756D1D39")  # type: ignore[arg-type]
        )

    assert gallery.retailer == "Amazon"
    assert gallery.image_urls == thumbnails
    assert "thumbnails" in gallery.source_description
    called_url = mock_fetch.call_args[0][0]
    assert "engine=amazon_product" in called_url
    assert "asin=B0756D1D39" in called_url
    assert "amazon_domain=amazon.in" in called_url


def test_extract_gallery_handles_gp_product_url_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(
        amazon_serpapi, "_fetch_json", return_value={"product_results": {"thumbnails": ["https://x/i.jpg"]}}
    ) as mock_fetch:
        gallery = _run(
            amazon_serpapi.extract_gallery(  # type: ignore[arg-type]
                None, "https://www.amazon.in/gp/product/B0756D1D39/ref=abc"
            )
        )
    assert gallery.image_urls == ["https://x/i.jpg"]
    assert "asin=B0756D1D39" in mock_fetch.call_args[0][0]


def test_extract_gallery_raises_on_unparseable_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    with pytest.raises(SerpApiSearchError):
        _run(amazon_serpapi.extract_gallery(None, "https://www.amazon.in/no-asin-in-this-url"))  # type: ignore[arg-type]


def test_extract_gallery_empty_thumbnails_returns_empty_gallery(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    with patch.object(amazon_serpapi, "_fetch_json", return_value={"product_results": {}}):
        gallery = _run(
            amazon_serpapi.extract_gallery(None, "https://www.amazon.in/dp/B0756D1D39")  # type: ignore[arg-type]
        )
    assert gallery.image_urls == []


def test_extract_gallery_captures_product_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `product_results.title` (the same API call already made for the
    gallery) is captured into `GalleryResult.product_title` - this is what
    lets verification distinguish a genuinely different product variant
    (e.g. "Charminar Long Grain, 1kg" vs the requested "Select Basmati")
    instead of treating the retailer's own listing title as unavailable.
    """
    _set_api_key(monkeypatch, "test-key")
    payload = {
        "product_results": {
            "title": "Charminar Select Basmati Rice 1 KG | Value Basmati rice",
            "thumbnails": ["https://m.media-amazon.com/images/I/img.jpg"],
        }
    }
    with patch.object(amazon_serpapi, "_fetch_json", return_value=payload):
        gallery = _run(
            amazon_serpapi.extract_gallery(None, "https://www.amazon.in/dp/B079GXGH83")  # type: ignore[arg-type]
        )
    assert gallery.product_title == "Charminar Select Basmati Rice 1 KG | Value Basmati rice"


def test_extract_gallery_missing_title_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, "test-key")
    payload = {"product_results": {"thumbnails": ["https://x/i.jpg"]}}
    with patch.object(amazon_serpapi, "_fetch_json", return_value=payload):
        gallery = _run(
            amazon_serpapi.extract_gallery(None, "https://www.amazon.in/dp/B0756D1D39")  # type: ignore[arg-type]
        )
    assert gallery.product_title is None


# --- API key handling ---------------------------------------------------------


def test_missing_api_key_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_key(monkeypatch, None)
    with pytest.raises(SerpApiInvalidApiKeyError):
        _run(amazon_serpapi.search_product_page(None, "q"))  # type: ignore[arg-type]


# --- error classification (direct unit tests of the translation helpers) -----


def test_raise_for_http_error_401_is_invalid_api_key() -> None:
    with pytest.raises(SerpApiInvalidApiKeyError):
        amazon_serpapi._raise_for_http_error(401, b'{"error": "Invalid API key"}')


def test_raise_for_http_error_429_is_rate_limit() -> None:
    with pytest.raises(SerpApiRateLimitError):
        amazon_serpapi._raise_for_http_error(429, b'{"error": "Rate limit exceeded"}')


def test_raise_for_http_error_other_status_is_generic() -> None:
    with pytest.raises(SerpApiSearchError):
        amazon_serpapi._raise_for_http_error(500, b'{"error": "Internal error"}')


def test_raise_for_error_payload_quota_exhausted_is_rate_limit() -> None:
    with pytest.raises(SerpApiRateLimitError):
        amazon_serpapi._raise_for_error_payload({"error": "You have run out of searches"})


def test_raise_for_error_payload_bad_key_is_invalid_api_key() -> None:
    with pytest.raises(SerpApiInvalidApiKeyError):
        amazon_serpapi._raise_for_error_payload({"error": "Invalid API key provided"})


def test_raise_for_error_payload_no_error_field_is_a_noop() -> None:
    amazon_serpapi._raise_for_error_payload({"organic_results": []})  # must not raise


# --- ASIN parsing --------------------------------------------------------------


def test_extract_asin_from_dp_url() -> None:
    assert amazon_serpapi._extract_asin("https://www.amazon.in/some-title/dp/B0756D1D39/ref=sr_1_1") == (
        "B0756D1D39"
    )


def test_extract_asin_from_gp_product_url() -> None:
    assert amazon_serpapi._extract_asin("https://www.amazon.in/gp/product/B0756D1D39") == "B0756D1D39"


def test_extract_asin_raises_when_not_found() -> None:
    with pytest.raises(SerpApiSearchError):
        amazon_serpapi._extract_asin("https://www.amazon.in/some/other/path")
