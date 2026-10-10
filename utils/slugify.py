"""Text-to-filesystem-safe-slug conversion."""

from __future__ import annotations

import re

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """
    Convert `text` into a lowercase, filesystem-safe slug: runs of
    non-alphanumeric characters become a single underscore, with leading/
    trailing underscores stripped.

    Example:
        >>> slugify("KISSAN FRESH TOMATO KETCHUP 950g PCH")
        'kissan_fresh_tomato_ketchup_950g_pch'
    """
    slug = _NON_SLUG_CHARS.sub("_", text.strip().lower()).strip("_")
    return slug or "product"
