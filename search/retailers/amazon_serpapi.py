"""
Amazon.in product search and gallery extraction via SerpApi - no Playwright.

This is the SerpApi-based replacement for `search/retailers/amazon.py`'s
Playwright scraping, wired in by `retailer_gallery_stage.py`'s `_RETAILERS`
registry. It honors the exact same shared contract every retailer module
implements (`search/retailers/__init__.py`):
`search_product_page(page, query) -> list[str]` and
`extract_gallery(page, product_url) -> GalleryResult`, both `async`. The
`page: Page` argument is accepted but never used - SerpApi calls are plain
authenticated HTTP requests, no browser needed. This mirrors the existing
precedent in `search/retailers/zepto.py`, whose functions likewise don't use
the shared `page` directly (it opens its own isolated context instead).

Two-call design, verified against SerpApi's live official documentation
(serpapi.com/amazon-search-api, serpapi.com/amazon-product-api) before
writing this:
    1. `search_product_page()` calls SerpApi's Amazon Search API
       (`engine=amazon`, `amazon_domain=amazon.in`) and reads
       `organic_results[]` for candidate product URLs (and their ASINs),
       skipping sponsored results via the documented `sponsored` boolean
       field - the SerpApi equivalent of the old Playwright module's
       "Sponsored" text-span check.
    2. `extract_gallery()` re-derives the ASIN from the candidate's product
       URL (Amazon URLs always embed it as `/dp/<ASIN>/` or
       `/gp/product/<ASIN>/` - the same `/dp/` assumption the old Playwright
       module already relied on for its link selector), then calls SerpApi's
       Amazon Product API (`engine=amazon_product`, `asin=...`) and reads
       `product_results.thumbnails[]` - documented by SerpApi as "Product
       image URLs" (plural, an array), and confirmed via a live example
       response on their docs page to return multiple full images (7, for
       one real product) rather than a single search-result thumbnail. This
       field is what makes it possible to retrieve the FULL gallery, not
       just the one small `organic_results[].thumbnail` string the search
       call alone would give.

Caveat carried forward rather than assumed away: SerpApi's documentation
does not explicitly guarantee `thumbnails[]` is a complete gallery (every
angle/variant) or full-resolution for every product - this should be
spot-checked live against real amazon.in listings (see HANDOFF.md /
tests/test_amazon_serpapi.py for how this was verified for this project).

`extract_gallery()` also reads `product_results.title` (the same API call
already made for the gallery, no extra request) into `GalleryResult.product_title`.
This exists because a live false-positive was found where the vision model
accepted a genuinely different product variant (a rice brand's "Long Grain"
line instead of the requested "Select Basmati" line) as just a naming-
convention difference, purely from the image and the ERP name - it had no
visibility into what Amazon itself calls the pictured product. Passing this
title through to `NemotronVisionModel.verify_product_match()` (see
`retailer_gallery_stage.py`'s `_try_candidate`) gives the verifier that
missing textual signal, without weakening or replacing the image check.

Reuses `search/serpapi.py`'s three public exception classes
(`SerpApiSearchError`, `SerpApiInvalidApiKeyError`, `SerpApiRateLimitError`)
so failures are typed consistently with the existing Google-Images SerpApi
integration, without modifying that file. The actual HTTP request/response
handling here is self-contained (not shared code) because that module's
translation logic lives in private instance methods tightly coupled to its
own class.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Final

from playwright.async_api import Page

from config import get_settings
from search.retailers import GalleryResult, dedupe_preserve_order
from search.serpapi import SerpApiInvalidApiKeyError, SerpApiRateLimitError, SerpApiSearchError

logger = logging.getLogger(__name__)

_SERPAPI_ENDPOINT: Final[str] = "https://serpapi.com/search.json"
_SEARCH_ENGINE: Final[str] = "amazon"
_PRODUCT_ENGINE: Final[str] = "amazon_product"
_AMAZON_DOMAIN: Final[str] = "amazon.in"

# Kept in sync with `retailer_gallery_stage._MAX_CANDIDATES_PER_RETAILER` -
# that's the actual cap the caller applies; collecting the same number here
# just avoids scanning every remaining organic result once enough candidates
# exist. Matches the old Playwright module's `_MAX_CANDIDATES` value too.
_MAX_CANDIDATES: Final[int] = 5

# Amazon ASINs are 10-character alphanumeric codes, always embedded in a
# product URL's path as either `/dp/<ASIN>/` or `/gp/product/<ASIN>/` - the
# same `/dp/` shape the old Playwright module's link selector
# (`a[href*="/dp/"]`) already relied on, so this is well-precedented, not a
# new guess.
_ASIN_PATTERN: Final[re.Pattern[str]] = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")


def _resolve_api_key() -> str:
    """
    Resolve the SerpAPI key fresh on every call (not cached at import/module
    scope) since these are plain functions, not a class instantiated once in
    `run.py` - there's no natural "construction time" to validate eagerly.
    """
    api_key = get_settings().serpapi_api_key
    if not api_key:
        raise SerpApiInvalidApiKeyError(
            "No SerpAPI key configured. Set the PIR_SERPAPI_API_KEY "
            "environment variable (see config.Settings.serpapi_api_key)."
        )
    return api_key


def _extract_asin(product_url: str) -> str:
    """Pull the ASIN out of an Amazon product URL, raising a clear error if it can't be found."""
    match = _ASIN_PATTERN.search(product_url)
    if not match:
        raise SerpApiSearchError(f"Could not extract an ASIN from Amazon product URL: {product_url!r}")
    return match.group(1)


async def _request(params: dict[str, str]) -> dict[str, Any]:
    """Run the blocking HTTP call off the event loop and translate failures."""
    url = f"{_SERPAPI_ENDPOINT}?{urllib.parse.urlencode(params)}"
    try:
        return await asyncio.to_thread(_fetch_json, url)
    except SerpApiSearchError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated to a domain error
        logger.exception("Unexpected failure calling SerpAPI (engine=%s)", params.get("engine"))
        raise SerpApiSearchError(f"Unexpected failure calling SerpAPI: {exc}") from exc


def _fetch_json(url: str) -> dict[str, Any]:
    """
    Blocking HTTP GET + JSON parse (invoked via `asyncio.to_thread`).
    Mirrors `search/serpapi.py`'s `_fetch_json`/error-translation pattern,
    self-contained here since that module's version is private instance
    methods tightly coupled to its own class.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    timeout = get_settings().request_timeout_seconds
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        _raise_for_http_error(exc.code, exc.read())
        raise  # unreachable - _raise_for_http_error always raises
    except OSError as exc:
        # Covers urllib.error.URLError, socket timeouts, DNS failures,
        # connection resets, etc. - all genuine network-layer failures.
        raise SerpApiSearchError(f"Network error contacting SerpAPI: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SerpApiSearchError(f"SerpAPI response was not valid JSON: {exc}") from exc

    _raise_for_error_payload(payload)
    return payload


def _raise_for_http_error(status_code: int, body: bytes) -> None:
    """Translate a non-2xx HTTP response into a specific domain exception."""
    message = _extract_error_message(body) or f"HTTP {status_code}"
    if status_code == 401:
        raise SerpApiInvalidApiKeyError(f"SerpAPI rejected the API key: {message}")
    if status_code == 429:
        raise SerpApiRateLimitError(f"SerpAPI rate/usage limit reached: {message}")
    raise SerpApiSearchError(f"SerpAPI request failed with HTTP {status_code}: {message}")


def _raise_for_error_payload(payload: dict[str, Any]) -> None:
    """
    SerpAPI can return HTTP 200 with an `"error"` field describing a failure
    (e.g. exhausted monthly quota) instead of a non-200 status; check for
    that explicitly since a 200 alone doesn't guarantee success.
    """
    error = payload.get("error") if isinstance(payload, dict) else None
    if not error:
        return

    lowered = str(error).lower()
    if "api key" in lowered:
        raise SerpApiInvalidApiKeyError(f"SerpAPI rejected the API key: {error}")
    if "rate" in lowered or "limit" in lowered or "run out of searches" in lowered:
        raise SerpApiRateLimitError(f"SerpAPI rate/usage limit reached: {error}")
    raise SerpApiSearchError(f"SerpAPI returned an error: {error}")


def _extract_error_message(body: bytes) -> str | None:
    """Best-effort extraction of an `"error"` field from an HTTP-error response body."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    return str(error) if error else None


async def search_product_page(page: Page, query: str) -> list[str]:
    """
    Search Amazon.in for `query` via SerpApi's Amazon Search API
    (`engine=amazon`) and return up to `_MAX_CANDIDATES` organic
    (non-sponsored) product URLs, in SerpApi's own ranked order, or `[]` if
    none were found. `page` is accepted for contract-parity with the other
    retailer modules but is not used - this call is a plain authenticated
    HTTP request, no browser involved.
    """
    params = {
        "engine": _SEARCH_ENGINE,
        "k": query,
        "amazon_domain": _AMAZON_DOMAIN,
        "api_key": _resolve_api_key(),
    }
    payload = await _request(params)

    organic_results = payload.get("organic_results") or []
    sponsored_count = 0
    candidates: list[str] = []
    for item in organic_results:
        if len(candidates) >= _MAX_CANDIDATES:
            break
        if not isinstance(item, dict):
            continue
        if item.get("sponsored"):
            sponsored_count += 1
            continue

        link = item.get("link_clean") or item.get("link")
        if isinstance(link, str) and link:
            candidates.append(link)

    deduped = dedupe_preserve_order(candidates)
    logger.info(
        "[Amazon/SerpApi] %d candidate(s) collected (skipped %d sponsored) for query=%r: %r",
        len(deduped),
        sponsored_count,
        query,
        deduped,
    )
    return deduped


async def extract_gallery(page: Page, product_url: str) -> GalleryResult:
    """
    Fetch the full image gallery for the Amazon product at `product_url` via
    SerpApi's Amazon Product API (`engine=amazon_product`), keyed by the
    ASIN parsed out of `product_url`. `page` is accepted for contract-parity
    but not used.
    """
    asin = _extract_asin(product_url)
    params = {
        "engine": _PRODUCT_ENGINE,
        "asin": asin,
        "amazon_domain": _AMAZON_DOMAIN,
        "api_key": _resolve_api_key(),
    }
    payload = await _request(params)

    product_results = payload.get("product_results") or {}
    thumbnails = product_results.get("thumbnails") or []
    image_urls = [url for url in thumbnails if isinstance(url, str) and url]

    title = product_results.get("title")
    product_title = title if isinstance(title, str) and title else None

    return GalleryResult(
        retailer="Amazon",
        product_page_url=product_url,
        image_urls=dedupe_preserve_order(image_urls),
        source_description="SerpApi Amazon Product API (engine=amazon_product) -> product_results.thumbnails[]",
        product_title=product_title,
    )
