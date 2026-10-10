"""
Product-page text for Shop & Compare: fetch a candidate's page and pull out
a short, bounded summary (page title, meta description, JSON-LD Product
name/brand/description, then visible text) for Nemotron to read as its
primary evidence.

Not a crawler: one GET per candidate URL, no link following, a short
timeout, a byte cap on the download and a character cap on the result.
`fetch_product_page_text` never raises - any failure returns `None` and the
caller falls back to the listing's title/retailer/price/image.

The URL comes from search results, so it is untrusted: fetching goes through
`utils.safe_http`, which refuses localhost, private, link-local and cloud-
metadata addresses (for the URL itself, every redirect hop and every address
the host resolves to), limits redirects, and caps response size and total
time. An unsafe URL is handled exactly like an unreachable one.

Search-engine result pages are not product pages and are never fetched.
Google Shopping's own `product_link` is one of these (a google.* results
page whose title mostly repeats the search query, which would bias the
model toward a match), so those candidates use the fallback. Extracted text
comes from the open web: it is passed to the model as data, never trusted
as instructions (see the prompt).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from typing import Any, Final

from utils import safe_http

logger = logging.getLogger(__name__)

MAX_PAGE_TEXT_CHARS: Final[int] = 700

_FETCH_TIMEOUT_SECONDS: Final[float] = 8.0
_TOTAL_FETCH_TIMEOUT_SECONDS: Final[float] = 15.0
_MAX_HTML_BYTES: Final[int] = 400_000
_USER_AGENT: Final[str] = "Mozilla/5.0 (compatible; ShopAndCompare/0.1)"

_MIN_VISIBLE_CHUNK_CHARS: Final[int] = 40
_WORD = re.compile(r"\w+")

_META_DESCRIPTION_KEYS: Final[tuple[str, ...]] = (
    "description",
    "og:description",
    "twitter:description",
)
# Content inside these tags is never product text.
_SKIPPED_TAGS: Final[frozenset[str]] = frozenset(
    {
        "script",
        "style",
        "noscript",
        "svg",
        "template",
        "nav",
        "header",
        "footer",
        "form",
        "button",
        "select",
        "iframe",
    }
)


class _ProductPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.meta: dict[str, str] = {}
        self.json_ld_blocks: list[str] = []
        self.visible_parts: list[str] = []
        self._in_title = False
        self._json_ld_buffer: list[str] | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): (value or "") for name, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (attributes.get("name") or attributes.get("property") or "").lower()
            if key in _META_DESCRIPTION_KEYS and attributes.get("content"):
                self.meta.setdefault(key, attributes["content"])
        elif tag == "script" and "ld+json" in attributes.get("type", "").lower():
            self._json_ld_buffer = []
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "script" and self._json_ld_buffer is not None:
            self.json_ld_blocks.append("".join(self._json_ld_buffer))
            self._json_ld_buffer = None
        if tag in _SKIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._json_ld_buffer is not None:
            self._json_ld_buffer.append(data)
        elif self._in_title:
            self.title_parts.append(data)
        elif self._skip_depth == 0 and data.strip():
            self.visible_parts.append(data)


def _clean(text: object) -> str:
    return " ".join(str(text).split()) if isinstance(text, (str, int, float)) else ""


def _find_products(node: Any) -> Iterator[dict[str, Any]]:
    """Every JSON-LD object typed `Product`, including inside lists and `@graph`."""
    if isinstance(node, list):
        for item in node:
            yield from _find_products(item)
    elif isinstance(node, dict):
        types = node.get("@type")
        if "Product" in (types if isinstance(types, list) else [types]):
            yield node
        yield from _find_products(node.get("@graph"))


def _json_ld_product_fields(blocks: list[str]) -> list[tuple[str, str]]:
    for block in blocks:
        try:
            data = json.loads(block)
        except ValueError:
            continue
        for product in _find_products(data):
            brand = product.get("brand")
            return [
                ("Product name", _clean(product.get("name"))),
                ("Brand", _clean(brand.get("name") if isinstance(brand, dict) else brand)),
                ("Product description", _clean(product.get("description"))),
            ]
    return []


def _words(text: str) -> set[str]:
    """Significant words (3+ characters), casefolded, for overlap checks."""
    return {word for word in _WORD.findall(text.casefold()) if len(word) >= 3}


def _visible_product_text(parts: list[str], about_words: set[str] | None) -> str:
    """
    Visible page text worth sending: sentence-length chunks only (short
    chunks are navigation, breadcrumbs and buttons - live Amazon pages
    yielded only category breadcrumbs, which misled the model), and, when
    the candidate's title is known, only chunks that mention it.
    """
    chunks = []
    for part in parts:
        chunk = _clean(part)
        if len(chunk) < _MIN_VISIBLE_CHUNK_CHARS:
            continue
        if about_words is not None and not (_words(chunk) & about_words):
            continue
        chunks.append(chunk)
    return " ".join(chunks)


def extract_product_text(
    html: str, max_chars: int = MAX_PAGE_TEXT_CHARS, about: str | None = None
) -> str:
    """
    A single-line summary of the product information in `html`, at most
    `max_chars` long: page title, JSON-LD Product name/brand/description,
    meta description, then visible page text with whatever room is left.
    Returns "" if nothing useful is found. Pure function.

    `about` is the candidate's listing title. When given, the page's
    title/description must share at least two significant words with it
    (one, if the title has only one); otherwise the page is not about the
    candidate - a generic site title from a script-rendered page, a robot
    check, an error page - and "" is returned so the caller falls back.
    This compares the page with the candidate's OWN title, never with the
    user's query, so it says nothing about whether the candidate matches.
    """
    parser = _ProductPageParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - malformed HTML: keep whatever was parsed so far
        logger.debug("[Product page] HTML parsing stopped early: %s", exc)

    about_words = _words(about) if about else None
    meta_description = next(
        (parser.meta[key] for key in _META_DESCRIPTION_KEYS if key in parser.meta), ""
    )
    fields = [
        ("Page title", _clean("".join(parser.title_parts))),
        *_json_ld_product_fields(parser.json_ld_blocks),
        ("Description", _clean(meta_description)),
    ]
    if about_words:
        shared = _words(" ".join(value for _, value in fields)) & about_words
        if len(shared) < min(2, len(about_words)):
            return ""
    fields.append(("Page text", _visible_product_text(parser.visible_parts, about_words)))

    parts: list[str] = []
    seen: set[str] = set()
    for label, value in fields:
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            parts.append(f"{label}: {value}")

    summary = " | ".join(parts)
    return summary if len(summary) <= max_chars else summary[: max_chars - 1].rstrip() + "…"


def is_product_page_url(url: str) -> bool:
    """A safe-to-fetch http(s) URL (see `utils.safe_http`) that is not a search-engine results page."""
    try:
        parts = safe_http.check_url(url)
    except safe_http.UnsafeURLError:
        return False
    return "google" not in (parts.hostname or "").lower().split(".")


def _fetch_html(url: str) -> str:
    """Blocking GET (run via `asyncio.to_thread`). Raises on any failure, unsafe URL or non-HTML response."""
    response = safe_http.fetch(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
        timeout=_FETCH_TIMEOUT_SECONDS,
        total_timeout=_TOTAL_FETCH_TIMEOUT_SECONDS,
        max_bytes=_MAX_HTML_BYTES,
        require_content_type="html",
    )
    return response.body.decode(response.charset or "utf-8", errors="replace")


async def fetch_product_page_text(
    url: str, max_chars: int = MAX_PAGE_TEXT_CHARS, about: str | None = None
) -> str | None:
    """
    Bounded product text from the page at `url`, or `None` if unavailable
    or (when `about`, the candidate's listing title, is given) not about
    that candidate - see `extract_product_text`. Never raises.
    """
    if not is_product_page_url(url):
        return None
    try:
        html = await asyncio.to_thread(_fetch_html, url)
    except Exception as exc:  # noqa: BLE001 - best-effort; the caller falls back to listing details
        logger.info("[Product page] could not fetch %s: %s", url, exc)
        return None
    return extract_product_text(html, max_chars, about) or None
