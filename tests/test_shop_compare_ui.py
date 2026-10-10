"""
Offline tests for the Shop & Compare result view in `app.py`.

The consumer sees only comparable offers (same_product /
same_product_different_size) and related alternatives. different_variant
and unclassified offers are never rendered as rows - only counted - but
stay in `ShoppingComparisonResult` for debugging. Streamlit is replaced by a
call recorder, so no Streamlit runtime, server, or network is involved. The
second half of this file exercises the real offer-card layout.
"""

from __future__ import annotations

from typing import Any

import pytest

import app
from models import (
    ClassifiedShoppingOffer,
    ShoppingComparisonResult,
    ShoppingMatchClassification,
    ShoppingMatchVerdict,
    ShoppingOffer,
)

_C = ShoppingMatchClassification


def _offer(title: str) -> ShoppingOffer:
    return ShoppingOffer(
        title=title,
        source="Some Store",
        price="Rs 10",
        thumbnail_url="https://example.com/thumb.jpg",
        product_link="https://www.google.com/shopping/somelisting",
    )


def _classified(title: str, classification: ShoppingMatchClassification) -> ClassifiedShoppingOffer:
    verdict = ShoppingMatchVerdict(classification=classification, confidence=0.9, reason="", model_name="fake")
    return ClassifiedShoppingOffer(offer=_offer(title), verdict=verdict)


def _result(
    classified: list[ClassifiedShoppingOffer] | None = None,
    unclassified: list[ShoppingOffer] | None = None,
    search_failed: bool = False,
) -> ShoppingComparisonResult:
    return ShoppingComparisonResult(
        product_name="Tata Salt 1 kg",
        offers=classified or [],
        unclassified_offers=unclassified or [],
        search_failed=search_failed,
    )


def _noisy_result() -> ShoppingComparisonResult:
    return _result(
        classified=[
            _classified("Tata Salt 1 kg", _C.SAME_PRODUCT),
            _classified("Tata Salt 1 kg Pack of 2", _C.SAME_PRODUCT_DIFFERENT_SIZE),
            _classified("Tata Lite 1 kg", _C.RELATED_PRODUCT),
            _classified("Silk Saree", _C.DIFFERENT_VARIANT),
            _classified("Wrist Watch", _C.DIFFERENT_VARIANT),
            _classified("Titanium Bar", _C.DIFFERENT_VARIANT),
        ],
        unclassified=[_offer("Unverified Listing A"), _offer("Unverified Listing B")],
    )


class _FakeStreamlit:
    """Records every `st.<name>(...)` call as `(name, args)`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        def _record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args))

        return _record

    def texts(self, name: str) -> list[str]:
        return [args[0] for call_name, args in self.calls if call_name == name and args]


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeStreamlit, list[tuple[str, str]]]:
    fake_st = _FakeStreamlit()
    rendered_rows: list[tuple[str, str]] = []
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(
        app,
        "_render_offer_row",
        lambda offer, color, label, **kwargs: rendered_rows.append((offer.title, label)),
    )
    return fake_st, rendered_rows


# --- grouping ------------------------------------------------------------------


def test_group_offers_splits_visible_from_counted() -> None:
    comparable, related, hidden, unverified = app._group_offers(_noisy_result())

    assert [c.offer.title for c in comparable] == ["Tata Salt 1 kg", "Tata Salt 1 kg Pack of 2"]
    assert [c.offer.title for c in related] == ["Tata Lite 1 kg"]
    assert hidden == 3
    assert unverified == 2


def test_different_variant_never_enters_comparable_or_related() -> None:
    comparable, related, _, _ = app._group_offers(_noisy_result())
    visible = comparable + related
    assert all(c.verdict.classification != _C.DIFFERENT_VARIANT for c in visible)


def test_group_offers_shows_ranked_deduplicated_order_without_unclassified() -> None:
    def priced(
        title: str, classification: ShoppingMatchClassification, price: float, source: str
    ) -> ClassifiedShoppingOffer:
        offer = _offer(title).model_copy(update={"extracted_price": price, "source": source})
        verdict = ShoppingMatchVerdict(classification=classification, confidence=0.9, reason="", model_name="fake")
        return ClassifiedShoppingOffer(offer=offer, verdict=verdict)

    result = _result(
        classified=[
            priced("Tata Lite 1 kg", _C.RELATED_PRODUCT, 39, "Zepto"),
            priced("Tata Salt 1 kg", _C.SAME_PRODUCT, 74, "amazon.in"),
            priced("Tata Salt 2 kg", _C.SAME_PRODUCT_DIFFERENT_SIZE, 55, "Zepto"),
            priced("Tata Salt 1 kg", _C.SAME_PRODUCT, 28, "Zepto"),
            priced("Tata Salt 1 kg", _C.SAME_PRODUCT, 80, "Amazon.in"),  # duplicate of the amazon.in listing
        ],
        unclassified=[_offer("Unverified Listing")],
    )

    comparable, related, hidden, unverified = app._group_offers(result)

    assert [(c.offer.title, c.offer.extracted_price) for c in comparable] == [
        ("Tata Salt 1 kg", 28),
        ("Tata Salt 1 kg", 74),
        ("Tata Salt 2 kg", 55),
    ]
    assert [c.offer.title for c in related] == ["Tata Lite 1 kg"]
    assert all(c.offer.title != "Unverified Listing" for c in comparable + related)
    assert (hidden, unverified) == (0, 1)
    assert len(result.offers) == 5  # the duplicate is still kept internally


def test_group_offers_keeps_hidden_offers_in_the_result_for_debugging() -> None:
    result = _noisy_result()
    app._group_offers(result)
    assert len(result.offers) == 6
    assert len(result.unclassified_offers) == 2


# --- count caption ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("hidden", "unverified", "expected"),
    [
        (13, 0, "13 unrelated listings hidden"),
        (1, 0, "1 unrelated listing hidden"),
        (0, 2, "2 offers couldn't be verified"),
        (0, 1, "1 offer couldn't be verified"),
        (13, 2, "13 unrelated listings hidden · 2 offers couldn't be verified"),
        (0, 0, None),
    ],
)
def test_hidden_counts_caption(hidden: int, unverified: int, expected: str | None) -> None:
    assert app._hidden_counts_caption(hidden, unverified) == expected


# --- rendering -------------------------------------------------------------------


def test_render_shows_only_comparable_and_related_rows_plus_counts(ui: Any) -> None:
    fake_st, rendered_rows = ui
    app._render_comparison_result(_noisy_result())

    assert [title for title, _ in rendered_rows] == ["Tata Salt 1 kg", "Tata Salt 1 kg Pack of 2", "Tata Lite 1 kg"]
    rendered_titles = {title for title, _ in rendered_rows}
    for never_shown in ("Silk Saree", "Wrist Watch", "Titanium Bar", "Unverified Listing A", "Unverified Listing B"):
        assert never_shown not in rendered_titles
    assert "3 unrelated listings hidden · 2 offers couldn't be verified" in fake_st.texts("caption")
    assert fake_st.texts("info") == []
    assert fake_st.texts("expander") == []


def test_render_empty_state_when_only_irrelevant_and_unverified_offers(ui: Any) -> None:
    fake_st, rendered_rows = ui
    result = _result(
        classified=[_classified(f"Noise {i}", _C.DIFFERENT_VARIANT) for i in range(13)],
        unclassified=[_offer("Unverified A"), _offer("Unverified B")],
    )
    app._render_comparison_result(result)

    assert rendered_rows == []
    assert fake_st.texts("info") == ["No matching offers found for this product."]
    assert "13 unrelated listings hidden · 2 offers couldn't be verified" in fake_st.texts("caption")


def test_render_empty_state_for_a_genuinely_empty_search(ui: Any) -> None:
    fake_st, rendered_rows = ui
    app._render_comparison_result(_result())

    assert rendered_rows == []
    assert fake_st.texts("info") == ["No matching offers found for this product."]
    assert fake_st.texts("caption") == []


def test_render_search_failure_is_not_an_empty_state(ui: Any) -> None:
    fake_st, rendered_rows = ui
    app._render_comparison_result(_result(search_failed=True))

    assert rendered_rows == []
    assert fake_st.texts("warning") == ["Couldn't check shopping offers right now. Please try again."]
    assert fake_st.texts("info") == []


def test_render_related_only_is_not_an_empty_state(ui: Any) -> None:
    fake_st, rendered_rows = ui
    app._render_comparison_result(_result(classified=[_classified("Saffola Muesli 500g", _C.RELATED_PRODUCT)]))

    assert [title for title, _ in rendered_rows] == ["Saffola Muesli 500g"]
    assert fake_st.texts("info") == []


def test_render_shows_no_matching_offers_when_only_an_accessory_was_found(ui: Any) -> None:
    """The Kissan case: the only match was an empty bottle, which the stage marks not relevant."""
    fake_st, rendered_rows = ui
    accessory = _classified("VELURA Mini Tomato Ketchup Bottle - 1 Empty Bottle", _C.DIFFERENT_VARIANT)

    app._render_comparison_result(_result(classified=[accessory]))

    assert rendered_rows == []
    assert fake_st.texts("info") == ["No matching offers found for this product."]
    assert "1 unrelated listing hidden" in fake_st.texts("caption")


# --- the real offer card ----------------------------------------------------------
# Layout: [ image ] title / retailer / classification tag   |   PRICE (large) + View offer button.
# Built from native containers only; a horizontal container wraps on a narrow screen,
# giving image -> title/retailer -> price -> button there.


class _Container:
    def __init__(self, fake: _RecordingStreamlit, container_id: int) -> None:
        self._fake = fake
        self.id = container_id

    def __enter__(self) -> _Container:
        self._fake.stack.append(self.id)
        return self

    def __exit__(self, *exc: object) -> bool:
        self._fake.stack.pop()
        return False


class _Call:
    def __init__(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], container: int | None) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.container = container  # id of the innermost container this call was made in


class _RecordingStreamlit:
    """A `st` stand-in that records every call, its keyword arguments, and which container it was made in."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self.stack: list[int] = []
        self.containers: dict[int, dict[str, Any]] = {}
        self.session_state: dict[str, Any] = {}

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_Call(name, args, kwargs, self.stack[-1] if self.stack else None))

    def container(self, *args: Any, **kwargs: Any) -> _Container:
        self._record("container", *args, **kwargs)
        container_id = len(self.containers) + 1
        self.containers[container_id] = kwargs
        return _Container(self, container_id)

    def tabs(self, labels: list[str]) -> list[_Container]:
        self._record("tabs", labels)
        return [self.container() for _ in labels]

    def text_input(self, *args: Any, **kwargs: Any) -> str:
        self._record("text_input", *args, **kwargs)
        return ""

    def button(self, *args: Any, **kwargs: Any) -> bool:
        self._record("button", *args, **kwargs)
        return False

    def __getattr__(self, name: str) -> Any:
        def _record(*args: Any, **kwargs: Any) -> None:
            self._record(name, *args, **kwargs)

        return _record

    def named(self, name: str) -> list[_Call]:
        return [call for call in self.calls if call.name == name]

    def in_container(self, **matching: Any) -> int:
        """The id of the one container created with exactly these keyword arguments."""
        found = [
            container_id
            for container_id, kwargs in self.containers.items()
            if all(kwargs.get(key) == value for key, value in matching.items())
        ]
        assert len(found) == 1, f"expected one container matching {matching}, found {len(found)}"
        return found[0]


@pytest.fixture
def card(monkeypatch: pytest.MonkeyPatch) -> _RecordingStreamlit:
    fake_st = _RecordingStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    return fake_st


def _priced_offer(
    title: str = "Tata Salt 1 kg",
    source: str | None = "amazon.in",
    price: str | None = "₹99",
) -> ShoppingOffer:
    return ShoppingOffer(
        title=title,
        source=source,
        price=price,
        thumbnail_url="https://example.com/thumb.jpg",
        product_link="https://www.amazon.in/dp/B000000000",
    )


def test_card_puts_image_first_and_price_with_button_in_a_right_aligned_column(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(), "green", "Same product")

    outer = card.in_container(border=True, horizontal=True)
    info = card.in_container(width="stretch")
    price_column = card.in_container(width="content", horizontal_alignment="right")
    assert card.containers[outer]["vertical_alignment"] == "center"

    outer_children = [call.name for call in card.calls if call.container == outer]
    assert outer_children[0] == "image"  # image on the left
    assert outer_children.count("container") == 2  # then the info column and the price column, in that order

    assert [call.name for call in card.calls if call.container == info] == ["markdown", "markdown", "badge"]
    # The price and the button live together in the right-hand column, price directly above the button.
    assert [call.name for call in card.calls if call.container == price_column] == ["subheader", "link_button"]


def test_price_is_a_large_heading_not_small_text(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(price="₹1,04,900"), "green", "Same product")

    (price,) = card.named("subheader")
    assert price.args == ("₹1,04,900",)
    assert price.kwargs["anchor"] is False
    assert card.named("caption") == []  # not buried in caption text


def test_related_alternative_card_has_a_smaller_price_and_a_quieter_button(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(), "blue", "Related alternative", compact=True)

    assert card.named("subheader") == []
    assert [call.args for call in card.named("markdown") if call.args[0].startswith("####")] == [("#### ₹99",)]
    assert card.named("link_button")[0].kwargs["type"] == "secondary"


def test_comparable_card_button_is_primary(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(), "green", "Same product")

    (button,) = card.named("link_button")
    assert button.args == ("View offer", "https://www.amazon.in/dp/B000000000")
    assert button.kwargs["type"] == "primary"


def test_card_without_a_price_says_so_instead_of_showing_a_blank(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(price=None), "green", "Same product")

    assert card.named("subheader") == []
    assert [call.args for call in card.named("caption")] == [("Price unavailable",)]
    assert len(card.named("link_button")) == 1  # the offer is still reachable


def test_classification_is_a_small_secondary_badge(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(), "orange", "Same product, different size")

    (badge,) = card.named("badge")
    assert badge.args == ("Same product, different size",)
    assert badge.kwargs["color"] == "orange"


@pytest.mark.parametrize(
    ("classification", "color", "label"),
    [
        (_C.SAME_PRODUCT, "green", "Same product"),
        (_C.SAME_PRODUCT_DIFFERENT_SIZE, "orange", "Same product, different size"),
        (_C.RELATED_PRODUCT, "blue", "Related alternative"),
    ],
)
def test_visible_classifications_keep_their_labels(
    classification: ShoppingMatchClassification, color: str, label: str
) -> None:
    assert app._CLASSIFICATION_DISPLAY[classification] == (color, label)


def test_image_has_one_bounded_width(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(), "green", "Same product")

    (image,) = card.named("image")
    assert image.args == ("https://example.com/thumb.jpg",)
    assert image.kwargs["width"] == app._THUMBNAIL_WIDTH == 96


def test_retailer_is_shown_by_its_plain_name(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(source="amazon.in"), "green", "Same product")

    retailer_lines = [call.args[0] for call in card.named("markdown") if "storefront" in call.args[0]]
    assert retailer_lines == [":material/storefront: Amazon"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("amazon.in", "Amazon"),
        ("Amazon.in", "Amazon"),
        ("JioMart Grocery", "JioMart"),
        ("DMart Ready", "DMart"),
        ("Instamart", "Swiggy Instamart"),
        ("Some Local Store", "Some Local Store"),
        (None, "Unknown seller"),
    ],
)
def test_display_retailer(source: str | None, expected: str) -> None:
    assert app._display_retailer(_priced_offer(source=source)) == expected


def test_no_raw_url_text_appears_on_the_card(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(), "green", "Same product")

    shown_text = [
        str(arg)
        for call in card.calls
        if call.name in {"markdown", "subheader", "caption", "badge"}
        for arg in call.args
    ]
    assert shown_text  # the card does show text, and none of it is a URL
    assert not any("http" in text for text in shown_text)


def test_long_title_is_cut_and_whitespace_collapsed(card: _RecordingStreamlit) -> None:
    long_title = "MAGGI 2-Minute Instant Noodles,   Masala  " + "Made With Choicest Quality Spices " * 10

    app._render_offer_row(_priced_offer(title=long_title), "green", "Same product")

    title_line = card.named("markdown")[0].args[0]
    assert title_line.startswith("**MAGGI 2-Minute Instant Noodles, Masala Made")
    assert title_line.endswith("…**")
    assert len(app._shorten_title(long_title)) <= app._MAX_TITLE_CHARS


def test_short_title_is_left_alone() -> None:
    assert app._shorten_title("Tata Salt 1 kg") == "Tata Salt 1 kg"
    assert app._shorten_title("x" * app._MAX_TITLE_CHARS) == "x" * app._MAX_TITLE_CHARS


@pytest.mark.parametrize(
    ("text", "escaped"),
    [
        ("Salt $5 and $6", r"Salt \$5 and \$6"),
        ("2*3_4", r"2\*3\_4"),
        ("[click](http://x)", r"\[click\]\(http://x\)"),
        ("# heading", r"\# heading"),
        ("plain ₹99", "plain ₹99"),
    ],
)
def test_listing_text_cannot_inject_markdown(text: str, escaped: str) -> None:
    assert app._escape_markdown(text) == escaped


def test_title_and_price_are_escaped_on_the_card(card: _RecordingStreamlit) -> None:
    app._render_offer_row(_priced_offer(title="Salt *Premium* $5", price="$829.00"), "green", "Same product")

    assert card.named("markdown")[0].args == (r"**Salt \*Premium\* \$5**",)
    assert card.named("subheader")[0].args == (r"\$829.00",)


# --- header, tabs, and grouping headings ---------------------------------------------


def test_shop_tab_goes_straight_to_the_search_box(card: _RecordingStreamlit) -> None:
    app._render_shopping_comparison_tab()

    # The page-level title and tagline live in main(); the tab does not repeat them.
    assert card.named("header") == []
    assert card.named("caption") == []
    assert card.named("text_input")[0].args[0] == "Product name"  # the existing search input is kept
    assert card.named("button")[0].args[0] == "Compare offers"


def test_page_is_branded_shop_and_compare(card: _RecordingStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_render_single_product_tab", lambda: None)
    monkeypatch.setattr(app, "_render_shopping_comparison_tab", lambda: None)

    app.main()

    (config,) = card.named("set_page_config")
    assert config.kwargs["page_title"] == "Shop & Compare"
    (title,) = card.named("title")
    assert title.args == ("Shop & Compare",)
    (tagline,) = card.named("caption")
    assert tagline.args == ("Compare prices from trusted retailers. Find the right product, faster.",)
    # Page config first, then title, tagline, tabs.
    assert [call.name for call in card.calls][:4] == ["set_page_config", "title", "caption", "tabs"]


def test_old_product_image_retriever_name_is_not_shown(card: _RecordingStreamlit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_render_single_product_tab", lambda: None)
    monkeypatch.setattr(app, "_render_shopping_comparison_tab", lambda: None)

    app.main()

    shown = " ".join(str(value).lower() for call in card.calls for value in (*call.args, *call.kwargs.values()))
    assert "image retriever" not in shown


def test_only_single_product_and_shop_and_compare_tabs_remain(
    card: _RecordingStreamlit, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(app, "_render_single_product_tab", lambda: rendered.append("single"))
    monkeypatch.setattr(app, "_render_shopping_comparison_tab", lambda: rendered.append("shop"))

    app.main()

    (tabs,) = card.named("tabs")
    assert tabs.args == (["Single product", "Shop & Compare"],)
    assert rendered == ["single", "shop"]


def test_batch_tab_code_is_gone_from_the_app() -> None:
    import inspect

    source = inspect.getsource(app).lower()
    assert "batch" not in source
    assert not hasattr(app, "_render_batch_tab")
    assert not hasattr(app, "_run_batch_pipeline")


def test_sections_are_headed_comparable_then_related_with_counts(
    card: _RecordingStreamlit, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        app,
        "_render_offer_row",
        lambda offer, color, label, compact=False: rows.append((offer.title, compact)),
    )

    app._render_comparison_result(_noisy_result())

    headings = [call.args[0] for call in card.named("markdown")]
    assert headings == ["**Comparable offers** · 2", "**Related alternatives** · 1"]
    assert "Not the exact product, but worth considering" in [call.args[0] for call in card.named("caption")]
    assert rows == [("Tata Salt 1 kg", False), ("Tata Salt 1 kg Pack of 2", False), ("Tata Lite 1 kg", True)]
