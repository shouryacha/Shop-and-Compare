"""
Domain models for Shop & Compare.

This module defines the data that flows *between* pipeline stages. It contains
no behavior beyond validation — every class here is a plain data container.

Design intent:
    - These models are the shared "vocabulary" of the whole system. Every
      interface in `interfaces.py` speaks in terms of these types, which is
      what lets search/download/classifier/selector/storage implementations
      be swapped out without touching `pipeline.py`.
    - Pydantic is used (not plain dataclasses) because several of these
      objects cross an I/O boundary (VLM JSON responses, HTTP results) and
      benefit from runtime validation + easy (de)serialization.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def _utc_now() -> datetime:
    """Naive UTC "now" (same value and serialized form as the deprecated `datetime.utcnow()`)."""
    return datetime.now(UTC).replace(tzinfo=None)


class ImageSide(str, Enum):
    """Which face of the product packaging an image shows."""

    FRONT = "front"
    BACK = "back"
    OTHER = "other"  # e.g. side panel, nutrition label, in-hand shot
    UNKNOWN = "unknown"  # VLM could not confidently determine the side


class SearchIntent(str, Enum):
    """
    The intent behind a generated search query.

    Kept separate from `ImageSide` because a query can be *aimed* at finding
    a front image without the returned candidate actually *being* one — the
    classifier is the sole source of truth for the final `ImageSide`.
    """

    GENERIC = "generic"
    FRONT = "front"
    BACK = "back"
    PACKSHOT = "packshot"  # e.g. "product name white background packshot"


class SearchQuery(BaseModel):
    """A single search query to be issued to one or more search providers."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., min_length=1, description="The literal query string.")
    intent: SearchIntent = Field(
        default=SearchIntent.GENERIC,
        description="Why this query was generated; used for provenance and debugging.",
    )


class ImageCandidate(BaseModel):
    """A single image URL returned by an ImageSearchProvider, not yet downloaded."""

    url: HttpUrl
    thumbnail_url: HttpUrl | None = None
    source_provider: str = Field(..., description="Registry key of the provider that found it.")
    source_query: SearchQuery
    page_url: HttpUrl | None = Field(
        default=None, description="The web page the image was embedded in, if known."
    )
    width: int | None = None
    height: int | None = None


class DownloadedImage(BaseModel):
    """An ImageCandidate that has been fetched and persisted to local disk."""

    candidate: ImageCandidate
    local_path: Path
    content_type: str | None = None
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None
    downloaded_at: datetime = Field(default_factory=_utc_now)


class ClassificationVerdict(BaseModel):
    """
    The Vision Language Model's judgement about a single downloaded image.

    This is the atomic unit of "intelligence" in the pipeline: everything
    upstream (search, download) is plumbing, and everything downstream
    (selection) is arithmetic over these verdicts.
    """

    is_correct_product: bool = Field(
        ..., description="Whether the image depicts the exact product queried for."
    )
    side: ImageSide
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's confidence in this verdict."
    )
    reasoning: str = Field(default="", description="Short natural-language justification.")
    raw_model_response: dict[str, Any] | str | None = Field(
        default=None, description="Unparsed response, kept for auditability/debugging."
    )
    model_name: str = Field(..., description="Registry key of the VisionModel that produced this.")


class ClassifiedImage(BaseModel):
    """A downloaded image paired with its classification verdict."""

    downloaded: DownloadedImage
    verdict: ClassificationVerdict


class ProductMatchVerdict(BaseModel):
    """
    The Vision Language Model's judgement about whether a single image
    (typically a retailer product page's first gallery image) actually
    depicts the requested product.

    Deliberately separate from `ClassificationVerdict`: that model's job is
    judging an image's `side` (front/back/other) for a candidate already
    presumed to be roughly the right product; this model's job is the prior
    question of whether a retailer *page* should be trusted at all - a
    different responsibility, with no notion of "side" involved.
    """

    is_match: bool = Field(
        ..., description="Whether this is clearly the same commercial product as requested."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's confidence in this verdict."
    )
    reason: str = Field(default="", description="Short natural-language justification.")
    brand: str | None = Field(
        default=None, description="Brand as read off the packaging, if legible."
    )
    product_name: str | None = Field(
        default=None, description="Product name/variant as read off the packaging, if legible."
    )
    weight: str | None = Field(
        default=None, description="Net weight/quantity as read off the packaging, if legible."
    )
    raw_model_response: dict[str, Any] | str | None = Field(
        default=None, description="Unparsed response, kept for auditability/debugging."
    )
    model_name: str = Field(..., description="Registry key of the VisionModel that produced this.")


class ProductImageResult(BaseModel):
    """
    Final output of the pipeline for one product.

    `front_image_path` / `back_image_path` are None when no candidate met
    the vision model's confidence floor for that side — callers must handle
    partial results rather than assuming both are always present.
    """

    product_name: str
    front_image_path: Path | None = None
    back_image_path: Path | None = None
    front_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    back_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_used: dict[str, str] = Field(
        default_factory=dict,
        description='Which provider resolved each role, e.g. {"search": "google_images", '
        '"vision": "gpt4v", "front_source": "amazon"}.',
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form extra context: timings, candidate counts, warnings, etc.",
    )
    created_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Shopping comparison (see shopping_comparison_stage.py)
# ---------------------------------------------------------------------------
# A separate, additive result shape from ProductImageResult above: that one
# describes a single winning image/gallery; this describes a LIST of
# shopping offers for one product, each independently labeled as the same
# product, the same product in a different size, or a different variant -
# deliberately never reduced to one winner, so the user can compare and
# choose. Cheap by design: offers carry only a remote thumbnail URL (never
# downloaded/persisted to `images/`) plus a click-through link.


class ShoppingMatchClassification(str, Enum):
    """
    How one shopping offer's product relates to the requested product.

    `RELATED_PRODUCT` is a meaningful alternative purchase: the same kind of
    consumer product, in a comparable form, for the same shopper use,
    differing only in brand, variant/line, and/or size (e.g. "Saffola
    Muesli" for a "Kellogg's Muesli" request, or "Tata Lite" for "Tata
    Salt"). Cross-brand alternatives are allowed; sharing only a brand
    (e.g. "Kellogg's Corn Flakes" for muesli) or only a generic word,
    broad category, ingredient, or adjacent use is not enough. Shown as a
    "related alternative", never mixed into the primary comparable-offers
    list.

    `DIFFERENT_VARIANT` is the "not relevant" bucket and the fail-closed
    default: anything that is neither the requested product nor a
    meaningful alternative to it. The UI hides these rows and shows only
    a count - see `app.py`. Despite its name it does NOT mean "a different
    variant of the requested product" (that is `RELATED_PRODUCT`); for that
    reason the prompt shows the model the label "not_relevant" for it, and
    the parser maps that back here (see `classifier/nemotron.py`).
    """

    SAME_PRODUCT = "same_product"
    SAME_PRODUCT_DIFFERENT_SIZE = "same_product_different_size"
    RELATED_PRODUCT = "related_product"
    DIFFERENT_VARIANT = "different_variant"


class ShoppingOffer(BaseModel):
    """
    One listing from a Google Shopping search result - not yet classified.

    `product_link` starts as Google's own listing page (Search API
    `shopping_results[].product_link`) - one hop from the actual retailer,
    not a direct merchant URL. For approved-retailer candidates,
    `shopping_comparison_stage.py` replaces it, before classification, with
    the resolved direct retailer URL when one is found (see
    `google_shopping_serpapi.resolve_direct_merchant_link`); it's left as
    Google's listing page when resolution fails or finds nothing. Callers
    never need to know which case applies - they just use `product_link`
    as the click-through target either way.
    """

    position: int | None = None
    title: str
    source: str | None = Field(
        default=None, description="Merchant/seller name, e.g. 'Amazon', 'Zepto'."
    )
    price: str | None = Field(default=None, description="Seller's own price string, e.g. '₹28'.")
    extracted_price: float | None = None
    thumbnail_url: HttpUrl
    product_link: HttpUrl = Field(
        description="Click-through link - see class docstring for what it points to and why."
    )
    page_token: str | None = Field(
        default=None,
        description="SerpApi's engine=google_immersive_product page_token for this exact listing "
        "(Search API shopping_results[].immersive_product_page_token), used to resolve a direct "
        "merchant link. None if the search response didn't include one for this offer.",
    )


class ShoppingMatchVerdict(BaseModel):
    """
    The Vision Language Model's classification of one ShoppingOffer against
    the requested product.

    `confidence` doubles as this verdict's 0.0-1.0 PRODUCT RELEVANCE score
    (not price attractiveness or purchase desirability - Nemotron is never
    asked to judge those) - a deliberate reuse rather than adding a
    duplicate field, since this codebase already treats `confidence` as the
    relevance-threshold signal for this exact use case.

    `evidence`/`brand_match`/`product_match`/`variant_match`/`size_match`
    are additive, OPTIONAL fields (all default to "unset") added to make
    the model's reasoning concrete and attribute-grounded rather than a
    single opaque sentence - see `prompts.SHOPPING_OFFER_CLASSIFICATION_SYSTEM_PROMPT`.
    They default to empty/`None` so every existing construction site
    (tests, older responses without these fields) keeps working unchanged.
    """

    classification: ShoppingMatchClassification
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's confidence in this verdict."
    )
    reason: str = Field(default="", description="Short natural-language justification.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Short, concrete, attribute-referencing statements supporting the classification "
        '(e.g. "Brand matches Kellogg\'s", "500g pack matches requested size").',
    )
    brand_match: bool | None = Field(
        default=None, description="Whether the candidate's brand matches."
    )
    product_match: bool | None = Field(
        default=None, description="Whether the candidate's product type/category matches."
    )
    variant_match: bool | None = Field(
        default=None, description="Whether the candidate's variant/flavor matches."
    )
    size_match: bool | None = Field(
        default=None, description="Whether the candidate's pack size matches."
    )
    model_name: str = Field(..., description="Registry key of the VisionModel that produced this.")


class ClassifiedShoppingOffer(BaseModel):
    """A ShoppingOffer paired with its classification verdict."""

    offer: ShoppingOffer
    verdict: ShoppingMatchVerdict


class ShoppingComparisonResult(BaseModel):
    """
    Final output of the shopping comparison stage for one product.

    `offers`: every offer that was successfully classified - all four
    classifications included, never filtered down to "the" answer; the UI
    decides how to group/display them.

    `unclassified_offers`: offers Google Shopping returned but that
    couldn't be classified for an infrastructure reason (a thumbnail
    download failure, or an NVIDIA Build request failing after retries) -
    NOT evidence of a product mismatch, and NOT treated as a match either.
    Preserved here rather than dropped; the UI reports them only as a
    count.

    `search_failed`: True when the Google Shopping search call itself
    failed (network timeout, HTTP/API error, SerpApi exception, malformed
    response) rather than genuinely returning zero results. Both lists
    empty AND `search_failed=False` means Google Shopping itself returned
    zero usable results - the only case that should read as "nothing
    found"; both lists empty AND `search_failed=True` means the search
    couldn't be completed at all and must be shown as a failure, never as
    a legitimate empty result.
    """

    product_name: str
    offers: list[ClassifiedShoppingOffer] = Field(default_factory=list)
    unclassified_offers: list[ShoppingOffer] = Field(default_factory=list)
    search_failed: bool = Field(
        default=False,
        description="True if the Google Shopping search call failed (not a genuine zero-result search).",
    )
    created_at: datetime = Field(default_factory=_utc_now)
