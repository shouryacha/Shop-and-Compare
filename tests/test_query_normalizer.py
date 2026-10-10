"""
Unit tests for `search/query_normalizer.py`'s `normalize_for_search()`.

Added alongside the "SEL" -> "Select" / "BSMTI" -> "Basmati" abbreviation
expansions, introduced after a live false-positive: the unexpanded query
"Kohinoor Charminar Sel Bsmti Rice 1KG PP" let Amazon's own search ranking
(via SerpApi) surface an unrelated "Charminar Long Grain" listing ahead of
the correct "Charminar Select Basmati Rice" one. See
`tests/test_amazon_serpapi.py` and `prompts.py` for the companion
candidate-title-verification fix.
"""

from __future__ import annotations

from search.query_normalizer import normalize_for_search


def test_sel_expands_to_select() -> None:
    assert normalize_for_search("Kohinoor Charminar Sel Basmati Rice 1kg") == (
        "Kohinoor Charminar Select Basmati Rice 1kg"
    )


def test_bsmti_expands_to_basmati() -> None:
    assert normalize_for_search("Kohinoor Bsmti Rice 1kg") == "Kohinoor Basmati Rice 1kg"


def test_sel_is_case_insensitive_and_whole_word_only() -> None:
    # Whole-word matching must not touch "Select" (already expanded) or a
    # word that merely contains "sel" as a substring.
    assert normalize_for_search("sel bsmti") == "Select Basmati"
    assert normalize_for_search("Vessel Bsmti") == "Vessel Basmati"


def test_full_kohinoor_regression_case() -> None:
    """
    The exact ERP string that originally exposed the false positive: noise
    word (PP) removed, both new abbreviations expanded, unit spelling
    canonicalized - all in the same pass, matching what
    `retailer_gallery_stage.py` actually sends to `search_product_page`.
    """
    result = normalize_for_search("Kohinoor Charminar Sel Bsmti Rice 1KG PP")
    assert result == "Kohinoor Charminar Select Basmati Rice 1kg"


def test_existing_abbreviations_still_expand() -> None:
    assert normalize_for_search("FUNFOODS MAYO COMBO 100g") == "FUNFOODS Mayonnaise 100g"


def test_existing_noise_words_still_removed() -> None:
    assert normalize_for_search("KISSAN KETCHUP 950g PP") == "KISSAN KETCHUP 950g"


def test_unrelated_words_are_left_alone() -> None:
    assert normalize_for_search("Brand Product Name 500g") == "Brand Product Name 500g"


# --- unit-separator preservation (see search/query_normalizer.py's
#     _normalize_units docstring - a live Google Shopping query-formulation
#     experiment found forcing the separator to empty hurt result relevance:
#     "Tata Salt 1kg" returned 2/10 relevant results vs "Tata Salt 1 kg"'s
#     10/10, for the identical underlying search engine and product) --------


def test_spaced_number_unit_keeps_its_space() -> None:
    assert normalize_for_search("Tata Salt 1 Kg") == "Tata Salt 1 kg"


def test_already_unspaced_number_unit_stays_unspaced() -> None:
    assert normalize_for_search("Tata Salt 1kg") == "Tata Salt 1kg"


def test_spaced_gram_variant_keeps_its_space() -> None:
    assert normalize_for_search("950 gms") == "950 g"


def test_spaced_litre_variant_keeps_its_space() -> None:
    assert normalize_for_search("2 ltrs") == "2 L"


def test_spaced_millilitre_variant_keeps_its_space() -> None:
    assert normalize_for_search("500 ML") == "500 ml"


def test_milligrams_are_not_in_the_unit_table_and_stay_untouched() -> None:
    assert normalize_for_search("500mg") == "500mg"
    assert normalize_for_search("500 mg") == "500 mg"


def test_already_working_products_unaffected_by_the_separator_fix() -> None:
    """
    Regression guard: the exact raw ERP strings for the three products
    already confirmed to get good Google Shopping results - none of them
    had a space before their unit, so the separator fix must be a no-op
    for all three.
    """
    assert normalize_for_search("KISSAN KETCHUP 950g") == "KISSAN KETCHUP 950g"
    assert normalize_for_search("Bournvita 5 Star Magic 500g") == "Bournvita 5 Star Magic 500g"
    assert normalize_for_search("Kohinoor Charminar Sel Bsmti Rice 1KG PP") == (
        "Kohinoor Charminar Select Basmati Rice 1kg"
    )


def test_spaced_unit_composes_correctly_with_abbreviation_and_noise_word_removal() -> None:
    """The separator fix must not interfere with the other three normalization steps."""
    assert normalize_for_search("FUNFOODS MAYO COMBO 100 g") == "FUNFOODS Mayonnaise 100 g"
