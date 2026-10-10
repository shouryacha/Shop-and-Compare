"""
Retailer-specific product-gallery extractors.

Each retailer module exposes two async functions of the shared shape
`search_product_page(page, query) -> list[str]` (up to a handful of
candidate product URLs, in the retailer's own ranked order, `[]` if none
found - the caller tries them in order until one is accepted, see
`retailer_gallery_stage.py`) and
`extract_gallery(page, product_url) -> GalleryResult`. There is
deliberately no shared base class beyond that return type: each site's
actual extraction mechanism is genuinely different, so forcing a common
abstraction over the mechanism itself would fight the sites rather than
describe them.

Active registry (`retailer_gallery_stage.py`): Amazon -> Google Shopping ->
Google Images (fallback) - `amazon_serpapi.py` and
`google_shopping_serpapi.py`, both entirely SerpApi-backed, no browser
involved for either. Every extractor here was built against a real API
response - none of the JSON paths/API fields are guessed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GalleryResult:
    """The outcome of one retailer extractor's attempt at one product page."""

    retailer: str
    product_page_url: str
    image_urls: list[str]
    source_description: str
    # The retailer's own listing title for this exact product, when the
    # extractor has one available (currently only `amazon_serpapi.py`,
    # sourced from SerpApi's Amazon Product API response). `None` for
    # retailers that don't surface one - existing callers/extractors that
    # don't pass this are unaffected. Used as additional textual evidence
    # during product-match verification (see `classifier/nemotron.py`),
    # never instead of the image.
    product_title: str | None = None


def dedupe_preserve_order(urls: list[str]) -> list[str]:
    """Return `urls` with duplicates removed, keeping the first occurrence of each."""
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        result.append(url)
    return result
