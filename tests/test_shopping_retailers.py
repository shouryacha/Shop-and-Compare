"""
Tests for `shopping_retailers.py`: normalizing Google Shopping `source`
names and exact alias matching against the approved-retailer allowlist.
Pure functions - no network involved.
"""

from __future__ import annotations

import pytest

from shopping_retailers import APPROVED_RETAILERS, canonical_retailer, normalize_source

_APPROVED = {
    "Amazon",
    "Flipkart",
    "Myntra",
    "Blinkit",
    "Zepto",
    "Nykaa",
    "BigBasket",
    "JioMart",
    "Swiggy Instamart",
    "DMart",
    "Tata CLiQ",
    "Croma",
    "Reliance Digital",
    "Vijay Sales",
    "AJIO",
    "FirstCry",
    "Purplle",
    "Tata 1mg",
    "Decathlon",
    "Apple",
    "Samsung",
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Amazon.in", "amazon"),
        ("amazon.in", "amazon"),
        ("  Amazon   India ", "amazon india"),
        ("www.croma.com", "croma"),
        ("reliancedigital.in", "reliancedigital"),
        ("Blinkit.com", "blinkit"),
        ("amazon.co.in", "amazon"),
        ("DMart Ready", "dmart ready"),
        ("Tata-CLiQ", "tata cliq"),
        ("JioMart   Grocery", "jiomart grocery"),
        ("1mg.com", "1mg"),
    ],
)
def test_normalize_source(raw: str, expected: str) -> None:
    assert normalize_source(raw) == expected


def test_approved_retailer_list_is_exactly_the_approved_set() -> None:
    assert set(APPROVED_RETAILERS) == _APPROVED


@pytest.mark.parametrize("retailer", sorted(_APPROVED))
def test_every_approved_retailer_matches_its_own_name(retailer: str) -> None:
    assert canonical_retailer(retailer) == retailer
    assert canonical_retailer(retailer.upper()) == retailer
    assert canonical_retailer(f"  {retailer.lower()}  ") == retailer


@pytest.mark.parametrize(
    ("source", "retailer"),
    [
        ("amazon.in", "Amazon"),
        ("Amazon.in", "Amazon"),
        ("Amazon India", "Amazon"),
        ("Amazon Fresh", "Amazon"),
        ("flipkart.com", "Flipkart"),
        ("Flipkart Grocery", "Flipkart"),
        ("myntra.com", "Myntra"),
        ("blinkit.com", "Blinkit"),
        ("zeptonow.com", "Zepto"),
        ("nykaa.com", "Nykaa"),
        ("bigbasket.com", "BigBasket"),
        ("Big Basket", "BigBasket"),
        ("JioMart Grocery", "JioMart"),
        ("JioMart Electronics", "JioMart"),
        ("jiomart.com", "JioMart"),
        ("Instamart", "Swiggy Instamart"),
        ("swiggy.com", "Swiggy Instamart"),
        ("DMart Ready", "DMart"),
        ("dmart.in", "DMart"),
        ("tatacliq.com", "Tata CLiQ"),
        ("Tata-CLiQ", "Tata CLiQ"),
        ("croma.com", "Croma"),
        ("reliancedigital.in", "Reliance Digital"),
        ("vijaysales.com", "Vijay Sales"),
        ("ajio.com", "AJIO"),
        ("firstcry.com", "FirstCry"),
        ("First Cry", "FirstCry"),
        ("purplle.com", "Purplle"),
        ("1mg", "Tata 1mg"),
        ("1mg.com", "Tata 1mg"),
        ("tata1mg.com", "Tata 1mg"),
        ("decathlon.in", "Decathlon"),
        ("Apple Store", "Apple"),
        ("apple.com", "Apple"),
        ("samsung.com", "Samsung"),
        ("Samsung Shop", "Samsung"),
    ],
)
def test_aliases_and_domain_forms_are_recognized(source: str, retailer: str) -> None:
    assert canonical_retailer(source) == retailer


@pytest.mark.parametrize(
    "source",
    [
        # Contain an approved name but are a different storefront or seller.
        "Shopsy By Flipkart",
        "Apple Premium Reseller",
        "Apple Premium Reseller - Unicorn",
        "Tata CLiQ Fashion",
        "Tata CLiQ Luxury",
        "Nykaa Fashion",
        "AJIO Luxe",
        "Samsung Authorized Reseller",
        "Croma Partner Store",
        "Amazonia Organics",
        # Brand-owned or import stores that are not on the list.
        "TATA Nutrikorner",
        "ubuy.co.in",
        "desertcart.in",
        # Unknown / local merchants.
        "icluster technologies",
        "iCluster Technologies",
        "Namo India Mart",
        "DIGI-किराना",
        "",
        None,
    ],
)
def test_non_approved_sellers_are_rejected(source: str | None) -> None:
    assert canonical_retailer(source) is None


def test_aliases_are_pre_normalized_and_unique() -> None:
    seen: dict[str, str] = {}
    for retailer, aliases in APPROVED_RETAILERS.items():
        assert normalize_source(retailer) in aliases, f"{retailer} does not match its own name"
        for alias in aliases:
            assert normalize_source(alias) == alias, f"{alias!r} ({retailer}) is not normalized"
            assert alias not in seen, f"{alias!r} maps to both {seen[alias]} and {retailer}"
            seen[alias] = retailer


# --- seller names seen in live Google Shopping results ---------------------------


@pytest.mark.parametrize(
    ("source", "retailer"),
    [
        ("FirstCry India", "FirstCry"),
        ("Myntra - MNow", "Myntra"),
        ("JioMart Grocery", "JioMart"),
        ("DMart Ready", "DMart"),
        ("Instamart", "Swiggy Instamart"),
        ("blinkit.com", "Blinkit"),
        ("bigbasket.com", "BigBasket"),
    ],
)
def test_live_seller_spellings_of_approved_retailers_are_recognized(
    source: str, retailer: str
) -> None:
    assert canonical_retailer(source) == retailer


@pytest.mark.parametrize(
    "source",
    [
        "Nykaa Man",
        "nykaaman.com",
        "TATA CLiQ LUXURY",
        "luxe.ajio.com",
        "StarQuik",
        "Hyperpure",
        "FarmLokal",
    ],
)
def test_live_sub_storefronts_and_other_sellers_stay_unapproved(source: str) -> None:
    assert canonical_retailer(source) is None
