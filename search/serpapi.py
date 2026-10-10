"""
SerpAPI Google Images search provider.

Implements `interfaces.ImageSearchProvider` by calling SerpAPI's Google
Images Search API (https://serpapi.com/search.json?engine=google_images) —
a documented, authenticated HTTP API, not a scraper.

Scope: this module performs ONLY search — it returns image URLs and neither
downloads image bytes nor classifies any image. It's the concrete search
provider `run.build_stage()` wires into `selection_stage.ProductImageSelectionStage`,
which is itself only reached as `retailer_gallery_stage.RetailerGalleryStage`'s
fallback, after Amazon and Google Shopping have both failed to produce an
accepted match.

Deliberately uses only the standard library for the HTTP call (`urllib`, run
off the event loop via `asyncio.to_thread`) rather than adding a new
third-party HTTP client dependency, since a single authenticated GET request
doesn't need one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Final

from config import get_settings
from interfaces import ImageSearchProvider
from models import ImageCandidate, SearchQuery

logger = logging.getLogger(__name__)

_PROVIDER_KEY: Final[str] = "serpapi"
_SERPAPI_ENDPOINT: Final[str] = "https://serpapi.com/search.json"
_ENGINE: Final[str] = "google_images"
_REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0


class SerpApiSearchError(Exception):
    """Raised when the SerpAPI request cannot produce a usable result."""


class SerpApiInvalidApiKeyError(SerpApiSearchError):
    """Raised when SerpAPI rejects the configured API key."""


class SerpApiRateLimitError(SerpApiSearchError):
    """Raised when SerpAPI reports the account's rate/usage limit was hit."""


class SerpApiImageSearchProvider(ImageSearchProvider):
    """`ImageSearchProvider` adapter backed by SerpAPI's Google Images engine."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Args:
            api_key: Explicit override, mainly for tests. Defaults to
                `config.Settings.serpapi_api_key` (populated from the
                `PIR_SERPAPI_API_KEY` environment variable).

        Raises:
            SerpApiSearchError: if no API key can be resolved from either
                source.
        """
        resolved_api_key = api_key or get_settings().serpapi_api_key
        if not resolved_api_key:
            raise SerpApiSearchError(
                "No SerpAPI key configured. Set the PIR_SERPAPI_API_KEY "
                "environment variable (see config.Settings.serpapi_api_key)."
            )
        self._api_key = resolved_api_key

    @property
    def name(self) -> str:
        return _PROVIDER_KEY

    async def search(self, query: SearchQuery, max_results: int) -> list[ImageCandidate]:
        """Satisfies `interfaces.ImageSearchProvider`; delegates to `search_images`."""
        urls = await self.search_images(query.text, max_results)
        return [
            ImageCandidate(url=url, source_provider=self.name, source_query=query) for url in urls
        ]

    async def search_images(self, query: str, max_results: int = 5) -> list[str]:
        """
        Return up to `max_results` original/full-size image URLs for `query`
        via SerpAPI's Google Images engine, in SerpAPI's ranked order.

        Results with no `"original"` image URL are skipped rather than
        substituted with a thumbnail. Returns an empty list if SerpAPI
        reports no image results for the query.

        Raises:
            ValueError: if `query` is empty/whitespace, or `max_results` is
                not positive.
            SerpApiInvalidApiKeyError: if SerpAPI rejects the API key.
            SerpApiRateLimitError: if the account's rate/usage limit was hit.
            SerpApiSearchError: for any other request, network, or response
                failure.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if max_results <= 0:
            raise ValueError("max_results must be a positive integer")

        logger.info("Querying SerpAPI Google Images for query=%r", query)
        payload = await self._request(query.strip())
        return self._extract_image_urls(payload, max_results)

    async def _request(self, query: str) -> dict[str, Any]:
        """Run the blocking HTTP call off the event loop and translate failures."""
        url = self._build_request_url(query)
        try:
            return await asyncio.to_thread(self._fetch_json, url)
        except SerpApiSearchError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a domain error
            logger.exception("Unexpected failure calling SerpAPI for query=%r", query)
            raise SerpApiSearchError(f"Unexpected failure calling SerpAPI: {exc}") from exc

    def _build_request_url(self, query: str) -> str:
        """Build the full SerpAPI request URL with query parameters."""
        params = {"engine": _ENGINE, "q": query, "api_key": self._api_key}
        return f"{_SERPAPI_ENDPOINT}?{urllib.parse.urlencode(params)}"

    def _fetch_json(self, url: str) -> dict[str, Any]:
        """
        Blocking HTTP GET + JSON parse (invoked via `asyncio.to_thread`).
        Isolated into its own method so the sync/async boundary lives in
        exactly one place.
        """
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            self._raise_for_http_error(exc.code, exc.read())
            raise  # unreachable - _raise_for_http_error always raises
        except OSError as exc:
            # Covers urllib.error.URLError, socket timeouts, DNS failures,
            # connection resets, etc. - all genuine network-layer failures.
            raise SerpApiSearchError(f"Network error contacting SerpAPI: {exc}") from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SerpApiSearchError(f"SerpAPI response was not valid JSON: {exc}") from exc

        self._raise_for_error_payload(payload)
        return payload

    def _raise_for_http_error(self, status_code: int, body: bytes) -> None:
        """Translate a non-2xx HTTP response into a specific domain exception."""
        message = self._extract_error_message(body) or f"HTTP {status_code}"

        if status_code == 401:
            raise SerpApiInvalidApiKeyError(f"SerpAPI rejected the API key: {message}")
        if status_code == 429:
            raise SerpApiRateLimitError(f"SerpAPI rate/usage limit reached: {message}")
        raise SerpApiSearchError(f"SerpAPI request failed with HTTP {status_code}: {message}")

    def _raise_for_error_payload(self, payload: dict[str, Any]) -> None:
        """
        SerpAPI can return HTTP 200 with an `"error"` field describing a
        failure (e.g. exhausted monthly quota) instead of a non-200 status;
        check for that explicitly since a 200 alone doesn't guarantee success.
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

    @staticmethod
    def _extract_error_message(body: bytes) -> str | None:
        """Best-effort extraction of an `"error"` field from an HTTP-error response body."""
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None
        error = parsed.get("error") if isinstance(parsed, dict) else None
        return str(error) if error else None

    def _extract_image_urls(self, payload: dict[str, Any], max_results: int) -> list[str]:
        """
        Pull up to `max_results` unique `"original"` image URLs out of
        SerpAPI's `images_results`, in order, skipping any result that has
        none.
        """
        image_results = payload.get("images_results")
        if not image_results:
            logger.info("SerpAPI returned no image results")
            return []

        seen: set[str] = set()
        urls: list[str] = []
        for item in image_results:
            if len(urls) >= max_results:
                break

            original = item.get("original") if isinstance(item, dict) else None
            if not original or not isinstance(original, str):
                logger.debug("Skipping SerpAPI result with no original image URL: %r", item)
                continue
            if original in seen:
                continue

            seen.add(original)
            urls.append(original)

        logger.info("Extracted %d unique original image URL(s) from SerpAPI", len(urls))
        return urls
