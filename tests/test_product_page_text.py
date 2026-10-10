"""
Tests for `product_page_text.py`: extracting bounded product text from a
page's HTML, and the fetch wrapper's fallbacks. The HTTP boundary
(`_fetch_html`) is mocked throughout - no network involved.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import product_page_text
from product_page_text import MAX_PAGE_TEXT_CHARS, extract_product_text, fetch_product_page_text

_PRODUCT_PAGE = """
<html>
  <head>
    <title>Buy Tata Salt 1 kg Online | Example Mart</title>
    <meta name="description" content="Tata Salt, vacuum evaporated iodised salt. 1 kg pouch.">
    <meta property="og:description" content="Should not be used when a plain description exists.">
    <style>.price { color: red; }</style>
    <script>window.tracking = {"brand": "IGNORED"};</script>
    <script type="application/ld+json">
      {"@context": "https://schema.org", "@type": "Product", "name": "Tata Salt Iodised 1 kg",
       "brand": {"@type": "Brand", "name": "Tata"},
       "description": "India's iodised salt, vacuum evaporated."}
    </script>
  </head>
  <body>
    <nav>Home Groceries Offers Login</nav>
    <h1>Tata Salt 1 kg</h1>
    <p>Net quantity: 1 kg. Free flowing iodised salt.</p>
    <footer>Copyright Example Mart</footer>
  </body>
</html>
"""


def _fetch(url: str, **kwargs: int) -> str | None:
    return asyncio.run(fetch_product_page_text(url, **kwargs))


# --- extraction ----------------------------------------------------------------


def test_extracts_title_meta_description_json_ld_and_visible_text() -> None:
    text = extract_product_text(_PRODUCT_PAGE)

    assert "Page title: Buy Tata Salt 1 kg Online | Example Mart" in text
    assert "Product name: Tata Salt Iodised 1 kg" in text
    assert "Brand: Tata" in text
    assert "Product description: India's iodised salt, vacuum evaporated." in text
    assert "Description: Tata Salt, vacuum evaporated iodised salt. 1 kg pouch." in text
    assert "Net quantity: 1 kg" in text


def test_scripts_styles_and_page_chrome_are_not_extracted() -> None:
    text = extract_product_text(_PRODUCT_PAGE)

    for unwanted in ("IGNORED", "color: red", "Login", "Copyright", "Should not be used"):
        assert unwanted not in text


def test_json_ld_product_is_found_inside_graph_and_lists() -> None:
    html = """
    <script type="application/ld+json">
      [{"@type": "BreadcrumbList"},
       {"@graph": [{"@type": "WebPage", "name": "Not the product"},
                   {"@type": ["Product", "Thing"], "name": "Kellogg's Muesli 500g", "brand": "Kellogg's",
                    "description": "Muesli with fruit and nuts."}]}]
    </script>
    """

    text = extract_product_text(html)

    assert "Product name: Kellogg's Muesli 500g" in text
    assert "Brand: Kellogg's" in text
    assert "Product description: Muesli with fruit and nuts." in text
    assert "Not the product" not in text


def test_malformed_json_ld_is_ignored_and_other_fields_still_extracted() -> None:
    html = '<title>Some Product</title><script type="application/ld+json">{not valid json</script>'

    assert extract_product_text(html) == "Page title: Some Product"


def test_meta_description_falls_back_to_open_graph() -> None:
    html = '<meta property="og:description" content="A 500 g pack of muesli.">'

    assert extract_product_text(html) == "Description: A 500 g pack of muesli."


def test_extracted_text_is_bounded() -> None:
    html = f"<title>Product</title><body><p>{'very long specification text ' * 2000}</p></body>"

    text = extract_product_text(html)

    assert len(text) <= MAX_PAGE_TEXT_CHARS
    assert text.endswith("…")
    assert len(extract_product_text(html, max_chars=120)) <= 120


def test_whitespace_is_collapsed_and_duplicate_fields_are_dropped() -> None:
    html = """
    <title>  Tata   Salt
        1 kg </title>
    <meta name="description" content="Tata Salt 1 kg">
    """

    assert extract_product_text(html) == "Page title: Tata Salt 1 kg"


@pytest.mark.parametrize("html", ["", "   ", "<html><body><script>var x = 1;</script></body></html>"])
def test_page_without_useful_text_yields_nothing(html: str) -> None:
    assert extract_product_text(html) == ""


# --- fetch wrapper ---------------------------------------------------------------


def test_fetch_returns_extracted_text_on_success() -> None:
    with patch.object(product_page_text, "_fetch_html", return_value=_PRODUCT_PAGE) as mock_fetch_html:
        text = _fetch("https://www.examplemart.in/p/tata-salt")

    mock_fetch_html.assert_called_once_with("https://www.examplemart.in/p/tata-salt")
    assert text is not None
    assert "Product name: Tata Salt Iodised 1 kg" in text


def test_fetch_failure_returns_none_instead_of_raising() -> None:
    with patch.object(product_page_text, "_fetch_html", side_effect=TimeoutError("timed out")):
        assert _fetch("https://www.examplemart.in/p/tata-salt") is None


def test_fetch_of_page_with_no_useful_text_returns_none() -> None:
    with patch.object(product_page_text, "_fetch_html", return_value="<html><body></body></html>"):
        assert _fetch("https://www.examplemart.in/p/tata-salt") is None


def test_fetch_respects_the_character_bound() -> None:
    with patch.object(product_page_text, "_fetch_html", return_value=_PRODUCT_PAGE):
        text = _fetch("https://www.examplemart.in/p/tata-salt", max_chars=60)

    assert text is not None
    assert len(text) <= 60


@pytest.mark.parametrize(
    "url",
    [
        "https://www.google.co.in/search?ibp=oshop&q=tata+salt",
        "https://www.google.com/shopping/product/123",
        "https://shopping.google.com/product/123",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "not a url",
    ],
)
def test_search_result_pages_and_non_http_urls_are_never_fetched(url: str) -> None:
    with patch.object(product_page_text, "_fetch_html") as mock_fetch_html:
        assert _fetch(url) is None

    mock_fetch_html.assert_not_called()


def test_retailer_domains_containing_the_word_google_elsewhere_are_still_fetched() -> None:
    with patch.object(product_page_text, "_fetch_html", return_value="<title>Product</title>") as mock_fetch_html:
        assert _fetch("https://www.googlestore-reseller.example/p/1") == "Page title: Product"

    mock_fetch_html.assert_called_once()


# --- the page must be about the candidate -----------------------------------------
# Found in live runs: JioMart's script-rendered pages return only the generic
# site title, and Amazon's visible text was only category breadcrumbs (which
# misled the model). `about` is the candidate's own listing title.


def test_generic_site_title_is_not_product_text() -> None:
    html = """<title>JioMart: India's online shopping</title>
    <meta name="description" content="JioMart: India's online shopping">"""

    assert extract_product_text(html, about="Tata Iodised Crystal Salt 1 kg") == ""


def test_robot_check_page_is_not_product_text() -> None:
    html = "<title>Robot Check</title><body><p>Enter the characters you see below to continue shopping today.</p></body>"

    assert extract_product_text(html, about="Maggi 2-Minute Instant Masala Noodles") == ""


def test_page_about_the_candidate_is_kept() -> None:
    html = "<title>Tata Salt, 2kg Poly Pack : Amazon.in: Grocery &amp; Gourmet Foods</title>"

    assert extract_product_text(html, about="Tata Salt") == (
        "Page title: Tata Salt, 2kg Poly Pack : Amazon.in: Grocery & Gourmet Foods"
    )


def test_single_shared_word_is_not_enough_for_a_multi_word_title() -> None:
    html = "<title>India's online shopping store</title>"

    assert extract_product_text(html, about="India Gate Basmati Rice 1 kg") == ""


def test_breadcrumbs_and_short_fragments_are_dropped_from_visible_text() -> None:
    html = """<title>Tata Salt, 1kg Pack</title>
    <body><a>Grocery</a><span>›</span><a>Salt &amp; Salt Substitutes</a><span>›</span><a>Rock Salt</a>
    <button>Add to Cart</button></body>"""

    text = extract_product_text(html, about="Tata Salt 1kg")

    assert text == "Page title: Tata Salt, 1kg Pack"
    assert "Rock Salt" not in text


def test_visible_sentences_about_the_candidate_are_kept_and_others_dropped() -> None:
    html = """<title>Tata Salt, 1kg Pack</title><body>
    <p>Tata Salt is vacuum evaporated iodised salt with a guaranteed iodine content.</p>
    <p>Get 5% back with the store credit card on every order above five hundred rupees.</p></body>"""

    text = extract_product_text(html, about="Tata Salt 1kg")

    assert "vacuum evaporated iodised salt" in text
    assert "credit card" not in text


def test_fetch_passes_the_candidate_title_through() -> None:
    generic = "<title>JioMart: India's online shopping</title>"
    with patch.object(product_page_text, "_fetch_html", return_value=generic):
        assert _fetch("https://www.jiomart.example/p/1", about="Tata Iodised Crystal Salt 1 kg") is None
        assert _fetch("https://www.jiomart.example/p/1") == "Page title: JioMart: India's online shopping"
