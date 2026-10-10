"""
Approved-retailer allowlist for Shop & Compare.

Only offers whose Google Shopping `source` is one of `APPROVED_RETAILERS`
enter the Shop & Compare pipeline; every other seller is discarded.

`source` is the merchant's display name, and the same retailer appears in
several spellings ("amazon.in", "Amazon.in", "Amazon India").
`canonical_retailer()` normalizes the name and looks it up in an explicit
alias table with EXACT matching only - never substring matching, which
would wrongly treat "Shopsy By Flipkart" as Flipkart, an "Apple Premium
Reseller" as Apple, or "Tata CLiQ Fashion" as Tata CLiQ.

To add a retailer or spelling, add its already-normalized alias (see
`normalize_source`) to `APPROVED_RETAILERS`. Domain forms need no alias of
their own when they normalize to an existing one ("blinkit.com" ->
"blinkit").
"""

from __future__ import annotations

import re
from typing import Final

APPROVED_RETAILERS: Final[dict[str, frozenset[str]]] = {
    "Amazon": frozenset({"amazon", "amazon india", "amazon fresh"}),
    "Flipkart": frozenset({"flipkart", "flipkart grocery", "flipkart minutes"}),
    "Myntra": frozenset({"myntra", "myntra mnow"}),
    "Blinkit": frozenset({"blinkit", "grofers"}),
    "Zepto": frozenset({"zepto", "zeptonow"}),
    "Nykaa": frozenset({"nykaa"}),
    "BigBasket": frozenset({"bigbasket", "big basket", "bb now"}),
    "JioMart": frozenset({"jiomart", "jiomart grocery", "jiomart electronics"}),
    "Swiggy Instamart": frozenset({"swiggy instamart", "instamart", "swiggy"}),
    "DMart": frozenset({"dmart", "dmart ready"}),
    "Tata CLiQ": frozenset({"tata cliq", "tatacliq"}),
    "Croma": frozenset({"croma"}),
    "Reliance Digital": frozenset({"reliance digital", "reliancedigital"}),
    "Vijay Sales": frozenset({"vijay sales", "vijaysales"}),
    "AJIO": frozenset({"ajio"}),
    "FirstCry": frozenset({"firstcry", "first cry", "firstcry india"}),
    "Purplle": frozenset({"purplle"}),
    "Tata 1mg": frozenset({"tata 1mg", "tata1mg", "1mg"}),
    "Decathlon": frozenset({"decathlon"}),
    "Apple": frozenset({"apple", "apple store"}),
    "Samsung": frozenset({"samsung", "samsung shop"}),
}

_ALIAS_TO_RETAILER: Final[dict[str, str]] = {
    alias: retailer for retailer, aliases in APPROVED_RETAILERS.items() for alias in aliases
}

# `.co.in` must be tried before `.in`.
_DOMAIN_SUFFIX = re.compile(r"\.(?:co\.in|com|in)$")
_NON_WORD_RUNS = re.compile(r"[\W_]+")


def normalize_source(source: str) -> str:
    """Casefold, drop a leading `www.` and a trailing `.co.in`/`.com`/`.in`, and collapse punctuation/whitespace."""
    text = source.casefold().strip()
    text = text.removeprefix("www.")
    text = _DOMAIN_SUFFIX.sub("", text)
    return _NON_WORD_RUNS.sub(" ", text).strip()


def canonical_retailer(source: str | None) -> str | None:
    """The approved retailer `source` exactly matches after normalization, or None if not approved."""
    if not source:
        return None
    return _ALIAS_TO_RETAILER.get(normalize_source(source))
