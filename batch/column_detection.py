"""
Column auto-detection for ERP Excel exports with unpredictable layouts.

Different customers export different column sets, in different orders, with
different header wording. Rather than assuming a fixed column position, each
header cell in the first row is normalized (lowercased, punctuation/whitespace
stripped) and matched against a curated set of known aliases for the two
required fields: EAN and Product Description. Every other column is ignored.
"""

from __future__ import annotations

import re
from typing import Any

# Normalized (see `_normalize_header`) aliases for each required field.
_EAN_ALIASES = {
    "ean",
    "eancode",
    "eannumber",
    "barcode",
    "gtin",
    "gtin13",
}
_DESCRIPTION_ALIASES = {
    "productdescription",
    "description",
    "productname",
    "itemdescription",
    "materialdescription",
    "product",
}

# Brand is OPTIONAL (unlike EAN/description above) - most ERP exports have
# it, but its absence is never an error, just no deterministic brand-mismatch
# checking for that file (see verification_overrides.py).
_BRAND_ALIASES = {
    "brand",
    "brandname",
    "manufacturer",
    "manufacturername",
}


class ColumnDetectionError(Exception):
    """Raised when the header row doesn't contain a recognizable EAN or Product Description column."""


def _normalize_header(value: Any) -> str:
    """Lowercase and strip everything but letters/digits, so 'EAN Code', 'ean_code', 'EAN-CODE' all match alike."""
    text = "" if value is None else str(value)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def detect_columns(header_row: list[Any]) -> tuple[int, int]:
    """
    Find the EAN and Product Description columns in `header_row` by matching
    normalized header text against known aliases. The first matching column
    (left to right) wins for each field.

    Returns:
        (ean_column_index, description_column_index), both 0-based.

    Raises:
        ColumnDetectionError: if either required column cannot be found.
    """
    ean_index: int | None = None
    description_index: int | None = None

    for index, raw_header in enumerate(header_row):
        normalized = _normalize_header(raw_header)
        if ean_index is None and normalized in _EAN_ALIASES:
            ean_index = index
        if description_index is None and normalized in _DESCRIPTION_ALIASES:
            description_index = index

    if ean_index is None:
        raise ColumnDetectionError(
            "Could not find an EAN column in the header row. Expected one of: "
            "EAN, EAN Code, EAN Number, Barcode, GTIN, GTIN-13 (or a similar variation)."
        )
    if description_index is None:
        raise ColumnDetectionError(
            "Could not find a Product Description column in the header row. Expected one of: "
            "Product Description, Description, Product Name, Item Description, "
            "Material Description, Product (or a similar variation)."
        )
    return ean_index, description_index


def detect_brand_column(header_row: list[Any]) -> int | None:
    """
    Find an optional Brand column, or `None` if the header row doesn't have
    one - never raises, since Brand is not a required field (see
    `_BRAND_ALIASES`'s own comment).
    """
    for index, raw_header in enumerate(header_row):
        if _normalize_header(raw_header) in _BRAND_ALIASES:
            return index
    return None
