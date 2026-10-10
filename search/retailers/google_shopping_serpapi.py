"""
Google Shopping product search and gallery extraction via SerpApi - no
Playwright. Second-priority active retailer, after Amazon
(`amazon_serpapi.py`), before the Google Images fallback (`selection_stage.py`).

Honors the same shared contract every retailer module implements
(`search/retailers/__init__.py`):
`search_product_page(page, query) -> list[str]` and
`extract_gallery(page, product_url) -> GalleryResult`, both `async`. The
`page: Page` argument is accepted but never used, exactly like
`amazon_serpapi.py` - these are plain authenticated HTTP requests, no
browser needed.

Two-call design, verified live against the real SerpApi endpoints before
writing this (not assumed from docs alone - `engine=google_product`, the
one the docs page describes, returned a live error: "The Google Product
service is no longer offered by Google."):
    1. `search_product_page()` calls SerpApi's Google Shopping Search API
       (`engine=google_shopping`) and reads `shopping_results[]` for
       candidates. Each candidate carries an `immersive_product_page_token`
       - the ONLY working way (confirmed live) to fetch that product's full
       detail/gallery is `engine=google_immersive_product` keyed by that
       token, not by `product_id` (the docs-described `engine=google_product`
       is dead).
    2. `extract_gallery()` re-derives the token from the candidate's
       `product_url` (see below for why it's encoded there) and calls
       `engine=google_immersive_product`, reading `product_results.thumbnails[]`
       - confirmed live to return a real multi-image gallery (10 images for
       one real product), and `product_results.title` for verification.

Why `product_url` is a SerpApi endpoint URL, not a human product page:
unlike Amazon (`/dp/<ASIN>/`), Google Shopping has no single canonical
per-item merchant page - `shopping_results[].product_link` points to a
Google-hosted aggregation/search page that does not embed the opaque
`immersive_product_page_token` needed for step 2, and the token itself
isn't a normal URL path segment. Returning the ready-made
`engine=google_immersive_product&page_token=...` URL (WITHOUT the API key -
added only at request time, never persisted) keeps the two-function
contract self-contained with no cross-call state, at the cost of
`retailer_product_url` in `metadata.json` / the "Open product page" UI link
not being a human-browsable page for this retailer specifically. Confirmed
live that these `gstatic.com` thumbnail URLs are plain, directly-downloadable
images (verified: real WEBP bytes, no browser/referrer needed) compatible
with the existing downloader and Nemotron's supported image types.

Reuses `search/serpapi.py`'s three public exception classes
(`SerpApiSearchError`, `SerpApiInvalidApiKeyError`, `SerpApiRateLimitError`)
for consistency, without modifying that file - see `amazon_serpapi.py` for
why the HTTP request/response handling itself is self-contained rather than
shared (that module's version is private instance methods tightly coupled
to its own class).

`search_shopping_offers()` (below `search_product_page`/`extract_gallery`)
is a THIRD, separate function for a different use case: the shopping-
comparison feature (`shopping_comparison_stage.py`) needs many candidate
listings with their prices/sellers/thumbnails for display, never a single
winner's full gallery - calling `extract_gallery`'s detail engine for every
candidate would cost one extra SerpApi credit per listing shown, for no
benefit (comparison only ever needs the cheap thumbnail, not a full
gallery). It reads `shopping_results[]` directly, in the same ONE
`engine=google_shopping` call `search_product_page` already makes - no new
engine, no new cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Final

from playwright.async_api import Page

from config import get_settings
from models import ShoppingOffer
from search.retailers import GalleryResult, dedupe_preserve_order
from search.serpapi import SerpApiInvalidApiKeyError, SerpApiRateLimitError, SerpApiSearchError
from shopping_retailers import canonical_retailer

logger = logging.getLogger(__name__)

_SERPAPI_ENDPOINT: Final[str] = "https://serpapi.com/search.json"
_SEARCH_ENGINE: Final[str] = "google_shopping"
_DETAIL_ENGINE: Final[str] = "google_immersive_product"
_GOOGLE_DOMAIN: Final[str] = "google.co.in"
_COUNTRY: Final[str] = "in"

# Kept in sync with `retailer_gallery_stage._MAX_CANDIDATES_PER_RETAILER` and
# the same constant in `amazon_serpapi.py` - the actual cap the caller
# applies; collecting the same number here just avoids building more
# candidate URLs than will ever be tried.
_MAX_CANDIDATES: Final[int] = 5

_PAGE_TOKEN_PARAM: Final[str] = "page_token"


def _resolve_api_key() -> str:
    """Resolved fresh on every call - see `amazon_serpapi._resolve_api_key` for why."""
    api_key = get_settings().serpapi_api_key
    if not api_key:
        raise SerpApiInvalidApiKeyError(
            "No SerpAPI key configured. Set the PIR_SERPAPI_API_KEY "
            "environment variable (see config.Settings.serpapi_api_key)."
        )
    return api_key


def _build_detail_url(page_token: str) -> str:
    """
    The safe, secret-free URL returned as a candidate's `product_url` - see
    module docstring for why this (not a human product page) is what's
    encoded here. The API key is added only inside `_request`, never here.
    """
    return f"{_SERPAPI_ENDPOINT}?{urllib.parse.urlencode({'engine': _DETAIL_ENGINE, _PAGE_TOKEN_PARAM: page_token})}"


def _extract_page_token(product_url: str) -> str:
    """Pull the `page_token` back out of a `product_url` built by `_build_detail_url`."""
    query = urllib.parse.urlsplit(product_url).query
    values = urllib.parse.parse_qs(query).get(_PAGE_TOKEN_PARAM)
    if not values or not values[0]:
        raise SerpApiSearchError(f"Could not extract a page_token from product URL: {product_url!r}")
    return values[0]


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
    """Blocking HTTP GET + JSON parse (invoked via `asyncio.to_thread`) - mirrors `amazon_serpapi._fetch_json`."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    timeout = get_settings().request_timeout_seconds
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        _raise_for_http_error(exc.code, exc.read())
        raise  # unreachable - _raise_for_http_error always raises
    except OSError as exc:
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
    SerpAPI can return HTTP 200 with an `"error"` field (e.g. exhausted
    quota, or a dead engine like `google_product` - confirmed live) instead
    of a non-200 status; check for that explicitly.
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
    Search Google Shopping (India) for `query` via SerpApi
    (`engine=google_shopping`) and return up to `_MAX_CANDIDATES` candidate
    "product URLs" - see module docstring for why these are
    `engine=google_immersive_product` endpoint URLs rather than human
    product pages - in SerpApi's own ranked order, or `[]` if none were
    found. `page` is accepted for contract-parity with the other retailer
    modules but is not used.
    """
    params = {
        "engine": _SEARCH_ENGINE,
        "q": query,
        "gl": _COUNTRY,
        "google_domain": _GOOGLE_DOMAIN,
        "api_key": _resolve_api_key(),
    }
    payload = await _request(params)

    shopping_results = payload.get("shopping_results") or []
    candidates: list[str] = []
    skipped_no_token = 0
    for item in shopping_results:
        if len(candidates) >= _MAX_CANDIDATES:
            break
        if not isinstance(item, dict):
            continue

        token = item.get("immersive_product_page_token")
        if not isinstance(token, str) or not token:
            skipped_no_token += 1
            continue

        candidates.append(_build_detail_url(token))

    deduped = dedupe_preserve_order(candidates)
    logger.info(
        "[Google Shopping/SerpApi] %d candidate(s) collected (skipped %d without a detail token) "
        "for query=%r",
        len(deduped),
        skipped_no_token,
        query,
    )
    return deduped


async def extract_gallery(page: Page, product_url: str) -> GalleryResult:
    """
    Fetch the full image gallery for the Google Shopping product at
    `product_url` via SerpApi's Google Immersive Product API
    (`engine=google_immersive_product`), keyed by the page token embedded in
    `product_url`. `page` is accepted for contract-parity but not used.
    """
    page_token = _extract_page_token(product_url)
    params = {
        "engine": _DETAIL_ENGINE,
        _PAGE_TOKEN_PARAM: page_token,
        "api_key": _resolve_api_key(),
    }
    payload = await _request(params)

    product_results = payload.get("product_results") or {}
    thumbnails = product_results.get("thumbnails") or []
    image_urls = [url for url in thumbnails if isinstance(url, str) and url]

    title = product_results.get("title")
    product_title = title if isinstance(title, str) and title else None

    return GalleryResult(
        retailer="Google Shopping",
        product_page_url=product_url,
        image_urls=dedupe_preserve_order(image_urls),
        source_description=(
            "SerpApi Google Immersive Product API (engine=google_immersive_product) "
            "-> product_results.thumbnails[]"
        ),
        product_title=product_title,
    )


async def search_shopping_offers(query: str, max_results: int = 10) -> list[ShoppingOffer]:
    """
    Search Google Shopping (India) for `query` and return up to
    `max_results` offers straight from `shopping_results[]` - title, price,
    seller, and a small thumbnail - with NO per-candidate detail-engine call
    here (see module docstring for why). `product_link` is Google's own
    listing page for that offer, not a direct merchant URL. Each offer's
    `immersive_product_page_token` (if present) is captured as
    `ShoppingOffer.page_token` - unused by this function, but it's what lets
    a caller selectively resolve a direct merchant link later, for only the
    offers worth the extra call (see `resolve_direct_merchant_link` and
    `shopping_comparison_stage.py`), without a second search.

    Entries missing a `title` or `thumbnail` (verified live: real responses
    occasionally have one without the other, e.g. a listing with no image
    yet indexed) are skipped, since a comparison row needs both to be useful.
    Live-verified: a single call to `engine=google_shopping` can return
    around 40 results for a common query - `max_results` caps how many are
    actually built into `ShoppingOffer`s.
    """
    params = {
        "engine": _SEARCH_ENGINE,
        "q": query,
        "gl": _COUNTRY,
        "google_domain": _GOOGLE_DOMAIN,
        "api_key": _resolve_api_key(),
    }
    payload = await _request(params)

    shopping_results = payload.get("shopping_results") or []
    offers: list[ShoppingOffer] = []
    skipped = 0
    for item in shopping_results:
        if len(offers) >= max_results:
            break
        if not isinstance(item, dict):
            continue

        title = item.get("title")
        thumbnail = item.get("thumbnail")
        product_link = item.get("product_link")
        if not (isinstance(title, str) and title and isinstance(thumbnail, str) and thumbnail):
            skipped += 1
            continue
        if not (isinstance(product_link, str) and product_link):
            skipped += 1
            continue

        page_token = item.get("immersive_product_page_token")
        try:
            offers.append(
                ShoppingOffer(
                    position=item.get("position"),
                    title=title,
                    source=item.get("source"),
                    price=item.get("price"),
                    extracted_price=item.get("extracted_price"),
                    thumbnail_url=thumbnail,
                    product_link=product_link,
                    page_token=page_token if isinstance(page_token, str) and page_token else None,
                )
            )
        except ValueError:
            # Malformed URL in a live field - skip this one offer rather
            # than failing the whole comparison over one bad listing.
            skipped += 1
            continue

    logger.info(
        "[Google Shopping/SerpApi] %d offer(s) collected (skipped %d incomplete/malformed) for query=%r",
        len(offers),
        skipped,
        query,
    )
    return offers


async def resolve_direct_merchant_link(page_token: str, source: str | None) -> str | None:
    """
    Resolve a direct merchant URL for ONE specific offer via SerpApi's
    Google Immersive Product API (`engine=google_immersive_product`), keyed
    by that offer's own `page_token` (`ShoppingOffer.page_token`).

    Reads `product_results.stores[]` (distinct from the `thumbnails[]`
    gallery `extract_gallery` reads elsewhere in this module) and returns
    the `link` of the first store that is the SAME approved retailer as
    `source`. Both names go through `shopping_retailers.canonical_retailer`
    (exact alias matching, never substring matching), so "Amazon.in"
    matches a store named "Amazon" but "Apple" does not match "Imagine
    Apple Premium Reseller".

    `source` must itself be an approved retailer: for any other seller no
    request is made at all.

    Deliberately never raises - returns `None` for every failure mode (no
    `page_token`, a `source` that is not an approved retailer, no matching
    store, an empty `stores[]`, or any request failure) so a caller can
    always fall back to the offer's existing `product_link` rather than
    treating this as an error.
    """
    retailer = canonical_retailer(source)
    if not page_token or retailer is None:
        return None

    try:
        params = {
            "engine": _DETAIL_ENGINE,
            _PAGE_TOKEN_PARAM: page_token,
            "api_key": _resolve_api_key(),
        }
        payload = await _request(params)
    except Exception as exc:  # noqa: BLE001 - best-effort; product_link is the caller's fallback
        logger.warning("[Google Shopping/SerpApi] direct-link resolution failed for source=%r: %s", source, exc)
        return None

    stores = (payload.get("product_results") or {}).get("stores") or []
    for store in stores:
        if not isinstance(store, dict):
            continue
        name = store.get("name")
        link = store.get("link")
        if isinstance(name, str) and canonical_retailer(name) == retailer and isinstance(link, str) and link:
            return link

    return None
