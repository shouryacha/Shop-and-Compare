"""
Query generation for Shop & Compare's search stage.

Responsibility: turn a raw product name into the search queries used to find
its front and back packaging images. This module performs no search itself —
no API calls, no scraping — it only produces query text.

Design intent:
    - `build_front_back_queries` is a small, pure, dependency-free function:
      given the same product name it always returns the same two strings,
      with no I/O. That makes it trivially testable and reusable anywhere
      (CLI, notebooks, other scripts), independent of the pipeline.
    - `SimpleQueryGenerator` adapts that pure function to
      `interfaces.QueryGenerator` so it can be injected into
      `selection_stage.ProductImageSelectionStage` exactly like any other
      query generator (e.g. a future LLM-based one).
"""

from __future__ import annotations

import logging

from interfaces import QueryGenerator
from models import SearchIntent, SearchQuery

logger = logging.getLogger(__name__)

_FRONT_SUFFIX = "front image"
_BACK_SUFFIX = "back image"


def build_front_back_queries(product_name: str) -> list[str]:
    """
    Return exactly two search-query strings for `product_name`: one aimed at
    finding the front-of-pack image, one at the back-of-pack image.

    The product name is used verbatim — not reformatted, re-cased, or
    altered beyond stripping surrounding whitespace — so downstream search
    providers see the exact identifying text (brand, variant, size, pack
    code) intact.

    Example:
        >>> build_front_back_queries("KISSAN FRESH TOMATO KETCHUP 950g PCH")
        ['KISSAN FRESH TOMATO KETCHUP 950g PCH front image', 'KISSAN FRESH TOMATO KETCHUP 950g PCH back image']

    Raises:
        ValueError: if `product_name` is empty or whitespace-only.
    """
    name = product_name.strip()
    if not name:
        raise ValueError("product_name must be a non-empty string")

    return [f"{name} {_FRONT_SUFFIX}", f"{name} {_BACK_SUFFIX}"]


class SimpleQueryGenerator(QueryGenerator):
    """
    `QueryGenerator` adapter that always produces exactly one front-image
    query and one back-image query via simple string concatenation — no LLM
    call, no API call, no ranking or scoring.
    """

    async def generate(self, product_name: str, max_queries: int) -> list[SearchQuery]:
        """
        Return the two `SearchQuery` objects built from
        `build_front_back_queries`, tagged with `SearchIntent.FRONT` and
        `SearchIntent.BACK` respectively.

        `max_queries` is accepted for `interfaces.QueryGenerator`
        compatibility but otherwise ignored: this generator always returns
        exactly two queries.
        """
        front_text, back_text = build_front_back_queries(product_name)
        logger.info("Generated 2 queries (front, back) for product=%r", product_name)

        return [
            SearchQuery(text=front_text, intent=SearchIntent.FRONT),
            SearchQuery(text=back_text, intent=SearchIntent.BACK),
        ]
