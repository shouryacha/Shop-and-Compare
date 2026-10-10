"""
Search-query normalization for retailer product-page search.

ERP/internal product names (what `retailer_gallery_stage.py` receives as
`product_name`) are written for inventory systems, not for how retailers
title their listings - they routinely carry internal packaging-type codes
("PCH", "PP", "PKT", "COMBO") that never appear on a retail page, and
abbreviate marketing terms ("MAYO") that retailers spell out in full. Both
hurt search recall without helping (or reflecting) what a customer would
actually search for.

This module ONLY improves the string sent to a retailer's own search box.
It is intentionally NOT used for verification:
`classifier.nemotron.NemotronVisionModel.verify_product_match` must keep receiving
the original, unmodified product name, so the actual correctness decision
is still made against the real ERP name, not a normalized approximation of
it.

Deliberately rule-based (fixed lookup tables + regex unit normalization),
not an LLM call: the one hard requirement here is "never invent a different
product," and a lookup-table substitution mechanically cannot add
information that wasn't already in the input - it can only remove known
noise tokens and expand known abbreviations, both from explicit, reviewable
lists. It also deliberately does NOT try to guess retailer-specific brand
phrasing (e.g. bridging "Gemini Dust Tea" to "Tata Tea Gemini") - that kind
of reasoning stays entirely with verification, which already handles it.
"""

from __future__ import annotations

import re

# Internal/inventory-only tokens that do not describe the marketed product
# and routinely don't appear on any retailer's page - safe to drop entirely
# for search purposes. Whole-word, case-insensitive. Extend this list as new
# ERP packaging/inventory codes are found; keep it to codes with no retail
# meaning - when in doubt, leave a word alone rather than add it here.
_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "COMBO",
        "PKT",
        "PCH",
        "PP",
        "CTN",
        "PET",
        "BTL",
    }
)

# Known ERP shorthand -> full marketed term. Whole-word, case-insensitive.
# This is the only way this module adds text that wasn't in the input, so
# it's kept short and reviewable - only add entries you're confident are
# unambiguous.
_ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "MAYO": "Mayonnaise",
    "CHOC": "Chocolate",
    "CHOCO": "Chocolate",
    "VEG": "Vegetable",
    "COND": "Condensed",
    "ORIG": "Original",
    "TOM": "Tomato",
    "SEL": "Select",
    "BSMTI": "Basmati",
}

# Unit spelling variants -> canonical form. Matches a number immediately
# followed (with or without whitespace) by any of these spellings, so it
# handles both attached ("950G", "200gm") and spaced ("950 gms") ERP styles.
_UNIT_CANONICAL: dict[str, str] = {
    "g": "g",
    "gm": "g",
    "gms": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "ml": "ml",
    "mls": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "ltr": "L",
    "ltrs": "L",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
}

_UNIT_PATTERN = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)(?P<sep>\s*)(?P<unit>"
    + "|".join(sorted(_UNIT_CANONICAL, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def _replace_whole_words(text: str, replacements: dict[str, str]) -> str:
    """Replace whole-word, case-insensitive matches found in `replacements`, leaving everything else as-is."""
    if not replacements:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(word) for word in replacements) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda m: replacements[m.group(0).upper()], text)


def _remove_whole_words(text: str, words: frozenset[str]) -> str:
    """Remove whole-word, case-insensitive matches found in `words`."""
    if not words:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(word) for word in words) + r")\b", re.IGNORECASE)
    return pattern.sub("", text)


def _normalize_units(text: str) -> str:
    """
    Rewrite `<number><any unit spelling>` to `<number><canonical unit>`,
    preserving whatever separator (none, or whitespace) was already between
    them, e.g. '950 gms' -> '950 g', '950gms' -> '950g'.

    The separator is preserved rather than forced to empty because a live
    Google Shopping query-formulation experiment found this matters for
    result relevance: "Tata Salt 1kg" (forced no-space) returned 2/10
    genuinely relevant results, while "Tata Salt 1 kg" (space preserved)
    returned 10/10 - the two engines this feeds (Amazon's search box,
    Google Shopping) are not naturally this format-sensitive.
    """

    def _replace(match: re.Match[str]) -> str:
        canonical_unit = _UNIT_CANONICAL[match.group("unit").lower()]
        return f"{match.group('number')}{match.group('sep')}{canonical_unit}"

    return _UNIT_PATTERN.sub(_replace, text)


def normalize_for_search(product_name: str) -> str:
    """
    Return a retailer-search-friendly version of `product_name`: known
    internal packaging/inventory codes removed, known abbreviations
    expanded, unit spelling canonicalized. Brand, flavour, variant, and
    weight are otherwise left exactly as given - only the fixed lists above
    are ever touched, and nothing is ever invented.

    This is for search only. Verification must keep using the original,
    unmodified `product_name` - see the module docstring.
    """
    text = product_name
    text = _replace_whole_words(text, _ABBREVIATION_EXPANSIONS)
    text = _remove_whole_words(text, _NOISE_WORDS)
    text = _normalize_units(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
