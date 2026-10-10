"""
Sanity tests for the domain models in `models.py`.

These only verify that the shared data layer is internally consistent
(construction, validation bounds, enum values). The retailer/verification/
fallback logic that uses these models is exercised through live,
real-API integration checks instead - see HANDOFF.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import (
    ImageCandidate,
    ImageSide,
    ProductImageResult,
    SearchIntent,
    SearchQuery,
)


def test_search_query_requires_nonempty_text() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(text="")


def test_search_query_defaults_to_generic_intent() -> None:
    query = SearchQuery(text="kissan ketchup 950g front")
    assert query.intent == SearchIntent.GENERIC


def test_image_candidate_requires_valid_url() -> None:
    query = SearchQuery(text="kissan ketchup 950g", intent=SearchIntent.FRONT)
    candidate = ImageCandidate(
        url="https://example.com/image.jpg",
        source_provider="google_images",
        source_query=query,
    )
    assert str(candidate.url).startswith("https://example.com/image.jpg")

    with pytest.raises(ValidationError):
        ImageCandidate(url="not-a-url", source_provider="google_images", source_query=query)


def test_product_image_result_allows_partial_results() -> None:
    result = ProductImageResult(product_name="KISSAN FRESH TOMATO KETCHUP 950g PCH")
    assert result.front_image_path is None
    assert result.back_image_path is None


def test_image_side_enum_values() -> None:
    assert {s.value for s in ImageSide} == {"front", "back", "other", "unknown"}
