"""
Streamlit UI for Shop & Compare.

Thin frontend only - contains no retrieval, verification, search, download,
or storage logic. It reuses:
    - `run.build_stage()` to construct the exact same `RetailerGalleryStage`
      (wired with its `ProductImageSelectionStage` fallback) that `run.py`'s
      CLI entry point uses.
    - `RetailerGalleryStage.run()` (unchanged) to actually perform the
      retrieval, via `asyncio.run(...)`, exactly like `run.py:main()` does.
    - `run.describe_result()` to interpret the finished `ProductImageResult`
      for display, so the "which mode is this result, what does it contain"
      logic is not duplicated between the CLI and this UI.
    - `run.build_shopping_stage()` / `ShoppingComparisonStage.compare()` for the
      Shop & Compare tab, which only renders the finished comparison.

Progress updates are surfaced by mirroring the log records the pipeline
already emits (via the `logging` calls already present in
`retailer_gallery_stage.py` / `selection_stage.py` / `classifier/nemotron.py` /
`storage/local_filesystem.py`) into two views on the Single product tab: a friendly,
checklist-style summary (pattern-matched from the exact log message text
already used in those modules) for normal users, and the full raw log inside
a "View logs" expander for debugging. No new logging calls were added to any business
module for this - this is pure presentation of output that already exists.

Run with: `streamlit run app.py`
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from classifier.nemotron import VisionModelError
from models import (
    ClassifiedShoppingOffer,
    ProductImageResult,
    ShoppingComparisonResult,
    ShoppingMatchClassification,
    ShoppingOffer,
)
from run import build_shopping_stage, build_stage, describe_result
from search.serpapi import SerpApiSearchError
from shopping_comparison_stage import rank_comparison_offers
from shopping_retailers import canonical_retailer
from storage.local_filesystem import LocalFilesystemStorage
from utils.logging_setup import configure_logging

configure_logging("INFO")

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_GALLERY_COLUMNS = 4

# --------------------------------------------------------------------------
# Friendly progress feed: maps the pipeline's EXISTING log message text
# (verbatim formats taken from retailer_gallery_stage.py, selection_stage.py,
# classifier/nemotron.py, storage/local_filesystem.py) to a short, human-readable
# line with a status icon. Purely presentational - if a message doesn't
# match anything here, it simply doesn't appear in the friendly feed (it's
# still available, unabridged, in the "View logs" expander).
# --------------------------------------------------------------------------
_FRIENDLY_LOG_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^Starting retailer-first retrieval"),
        "\U0001f50d Searching retailers for a matching product",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] trying candidate (?P<idx>\d+)/(?P<count>\d+):"),
        "\U0001f50e Checking candidate {idx}/{count} on {retailer}",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] found \d+ gallery image"),
        "✅ Extracted the image gallery from {retailer}",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] accepted"),
        "✅ {retailer} accepted - downloading the full gallery",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] rejected:"),
        "❌ {retailer} rejected - trying the next retailer",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] no product page found"),
        "❌ No product page found on {retailer}",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] no gallery images found"),
        "❌ No image gallery found on {retailer}",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] (native )?search failed"),
        "⚠️ Search failed on {retailer}",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] failed to open/extract product page"),
        "⚠️ Could not open the product page on {retailer}",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] could not download the first gallery image"),
        "⚠️ Could not download an image from {retailer}",
    ),
    (
        re.compile(r"^\[(?P<retailer>[^\]]+)\] product-match verification failed"),
        "⚠️ Verification error on {retailer} - trying the next retailer",
    ),
    (
        re.compile(r"^No retailer produced an accepted product page"),
        "↩️ No retailer matched - falling back to Google Images",
    ),
    (re.compile(r"^Starting image selection stage"), "\U0001f50d Searching Google Images"),
    (
        re.compile(r"^Comparing \d+ candidate\(s\) for product=.* side=(?P<side>\w+)"),
        "\U0001f504 Comparing {side} candidates with the vision model",
    ),
    (
        re.compile(r"^Selected candidate index=.* for side=(?P<side>\w+)"),
        "✅ Selected the best {side} image",
    ),
    (
        re.compile(r"^Verifying product match for image="),
        "\U0001f504 Verifying the product with the vision model",
    ),
    (re.compile(r"^Persisted (?P<side>front|back) image"), "✅ Saved the {side} image"),
    (re.compile(r"^Persisted \d+ gallery image"), "✅ Saved the gallery images"),
    (re.compile(r"^Wrote metadata for"), "✅ Wrote metadata.json"),
]


def _friendly_message(record: logging.LogRecord) -> str | None:
    """Return a short, friendly progress line for `record`, or `None` if it has no friendly mapping."""
    text = record.getMessage()
    for pattern, template in _FRIENDLY_LOG_PATTERNS:
        match = pattern.search(text)
        if match:
            return template.format(**match.groupdict())
    return None


class _StreamlitProgressHandler(logging.Handler):
    """
    Mirrors log records the pipeline already emits into Streamlit
    placeholders: the raw log, and the friendly, pattern-matched checklist.
    """

    def __init__(
        self, friendly_placeholder: DeltaGenerator, raw_placeholder: DeltaGenerator
    ) -> None:
        super().__init__()
        self._friendly_placeholder = friendly_placeholder
        self._raw_placeholder = raw_placeholder
        self._friendly_lines: list[str] = []
        self._raw_lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self._raw_lines.append(self.format(record))
        self._raw_placeholder.code("\n".join(self._raw_lines[-300:]), language="text")

        friendly = _friendly_message(record)
        if friendly:
            self._friendly_lines.append(friendly)
            self._friendly_placeholder.markdown(
                "\n".join(f"- {line}" for line in self._friendly_lines)
            )


def _run_pipeline(
    product_name: str, erp_brand: str | None = None
) -> tuple[ProductImageResult, LocalFilesystemStorage] | None:
    """
    Build the stage, run it with a live checklist/log view, and return
    `(result, storage)` or `None`. `erp_brand` is optional - used only for
    `verification_overrides.py`'s deterministic brand-mismatch check, never
    for search or sent to the vision model (see `RetailerGalleryStage.run`).
    """
    try:
        stage, storage = build_stage()
    except (SerpApiSearchError, VisionModelError, ValueError) as exc:
        st.error(f"Could not initialize a required provider: {exc}")
        return None

    with st.status("Running retrieval...", expanded=True) as status:
        friendly_placeholder = st.empty()
        with st.expander("View logs", icon=":material/terminal:"):
            raw_placeholder = st.empty()

        handler = _StreamlitProgressHandler(friendly_placeholder, raw_placeholder)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            result = asyncio.run(stage.run(product_name, erp_brand=erp_brand))
        except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the UI
            status.update(label="Retrieval failed", state="error", expanded=True)
            st.error(f"Retrieval failed: {exc}")
            return None
        finally:
            root_logger.removeHandler(handler)

        status.update(label="Completed", state="complete", expanded=False)

    return result, storage


@st.dialog("Image preview", width="large")
def _preview_dialog(paths: list[Path]) -> None:
    """Large, focused view of one gallery image, with Previous/Next navigation - the closest native equivalent to a lightbox."""
    index = st.session_state.get("gallery_preview_index", 0)
    index = max(0, min(index, len(paths) - 1))
    path = paths[index]

    st.image(str(path), width="stretch")
    st.caption(f"{path.name} · image {index + 1} of {len(paths)}")

    with st.container(horizontal=True):
        if st.button("Previous", icon=":material/chevron_left:", disabled=index == 0):
            st.session_state["gallery_preview_index"] = index - 1
            st.rerun(scope="fragment")
        if st.button("Next", icon=":material/chevron_right:", disabled=index == len(paths) - 1):
            st.session_state["gallery_preview_index"] = index + 1
            st.rerun(scope="fragment")


def _render_gallery(image_paths: list[Path]) -> None:
    """Responsive grid of gallery images; clicking a thumbnail's button opens a full-size dialog preview."""
    existing_paths = [p for p in image_paths if p.exists()]

    st.subheader(f"Gallery ({len(existing_paths)})")
    if not existing_paths:
        st.warning("No images were found for this product.")
        return

    for row_start in range(0, len(existing_paths), _GALLERY_COLUMNS):
        row_paths = existing_paths[row_start : row_start + _GALLERY_COLUMNS]
        columns = st.columns(_GALLERY_COLUMNS)
        for offset, (column, path) in enumerate(zip(columns, row_paths)):
            global_index = row_start + offset
            with column:
                with st.container(border=True):
                    st.image(str(path), width="stretch")
                    st.caption(path.name)
                    if st.button(
                        "View full size",
                        key=f"preview_btn_{global_index}",
                        icon=":material/zoom_in:",
                        width="stretch",
                    ):
                        st.session_state["gallery_preview_index"] = global_index
                        _preview_dialog(existing_paths)


def _render_summary(result: ProductImageResult, info: dict[str, Any]) -> None:
    """Clean summary cards: product name, retailer used, verification confidence, gallery count, product URL."""
    if info["mode"] == "retailer_gallery":
        retailer_label = info["retailer"]
        confidence = result.metadata.get("verification_confidence")
        confidence_label = "Verification confidence"
    elif info["mode"] == "no_result_fallback_disabled":
        retailer_label = "None (fallback disabled)"
        confidence = None
        confidence_label = "Verification confidence"
    else:
        retailer_label = "Google Images (fallback)"
        side_confidences = [
            c for c in (result.front_confidence, result.back_confidence) if c is not None
        ]
        confidence = sum(side_confidences) / len(side_confidences) if side_confidences else None
        confidence_label = "Verification confidence (avg)"

    confidence_display = f"{confidence:.0%}" if confidence is not None else "N/A"

    st.subheader("Summary")
    with st.container(horizontal=True):
        st.metric("Product name", result.product_name, border=True)
        st.metric("Retailer used", retailer_label, border=True)
        st.metric(confidence_label, confidence_display, border=True)
        st.metric("Gallery image count", str(len(info["image_paths"])), border=True)

    if info["mode"] == "retailer_gallery" and info["retailer_url"]:
        st.link_button("Open product page", info["retailer_url"], icon=":material/open_in_new:")


def _render_result(result: ProductImageResult, storage: LocalFilesystemStorage) -> None:
    """Display the summary, gallery, and (collapsed) full metadata.json for a finished result."""
    info = describe_result(result)

    if info["mode"] == "no_result_fallback_disabled":
        st.warning(
            "No result - Amazon and Google Shopping both failed, and Google/SerpAPI fallback is "
            "temporarily disabled (see ENABLE_GOOGLE_FALLBACK in retailer_gallery_stage.py)."
        )
    else:
        st.success("Retrieval complete.")
    _render_summary(result, info)
    st.space("medium")
    _render_gallery(info["image_paths"])

    metadata_path = storage.product_dir(result.product_name) / "metadata.json"
    with st.expander("Advanced details (metadata.json)", icon=":material/settings:"):
        if metadata_path.exists():
            st.json(json.loads(metadata_path.read_text(encoding="utf-8")))
        else:
            st.warning(f"metadata.json not found at {metadata_path}")


def _render_single_product_tab() -> None:
    st.caption("Find verified, high-resolution product images from major retailers")

    with st.container(horizontal=True, vertical_alignment="bottom"):
        product_name = st.text_input(
            "Product name",
            placeholder="e.g. KISSAN FRESH TOMATO KETCHUP 950g PCH",
            width="stretch",
        )
        erp_brand = st.text_input(
            "Brand (optional)",
            placeholder="e.g. JERSEY",
            help=(
                "Only used for a deterministic post-verification check: if the vision model "
                "reads a clearly different, unrelated brand off the retailer image, that candidate "
                "is rejected regardless of the model's own is_match. Leave blank to skip this check "
                "(the model's own verdict is then used as-is, exactly as before)."
            ),
        )
        retrieve_clicked = st.button("Retrieve images", type="primary", icon=":material/search:")

    if retrieve_clicked:
        cleaned_name = product_name.strip()
        if not cleaned_name:
            st.error("Product name cannot be empty.")
        else:
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_storage", None)
            outcome = _run_pipeline(cleaned_name, erp_brand=erp_brand.strip() or None)
            if outcome is not None:
                st.session_state["last_result"], st.session_state["last_storage"] = outcome

    if "last_result" in st.session_state:
        _render_result(st.session_state["last_result"], st.session_state["last_storage"])


# ---------------------------------------------------------------------------
# Shop & Compare tab - separate from the image-retrieval tab above. Reuses
# `run.build_shopping_stage()` / `shopping_comparison_stage.ShoppingComparisonStage`
# exactly as that tab reuses `run.build_stage()` - no retrieval/classification
# logic lives here, only display. Deliberately a clean comparison view, not a
# technical debug interface: no raw logs or metadata expanders - this is meant
# for an end user comparing purchase options, not for debugging the pipeline.
#
# Each offer is a card, built only from native Streamlit layout (no custom
# CSS): a horizontal bordered container whose children wrap onto the next line
# on a narrow screen, so a phone gets image -> title/retailer -> price -> button.
# ---------------------------------------------------------------------------

# Badge colour and label for each classification. The label is a small,
# secondary tag on the card - the price is the dominant element.
_BadgeColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]

_CLASSIFICATION_DISPLAY: dict[ShoppingMatchClassification, tuple[_BadgeColor, str]] = {
    ShoppingMatchClassification.SAME_PRODUCT: ("green", "Same product"),
    ShoppingMatchClassification.SAME_PRODUCT_DIFFERENT_SIZE: (
        "orange",
        "Same product, different size",
    ),
    ShoppingMatchClassification.RELATED_PRODUCT: ("blue", "Related alternative"),
    ShoppingMatchClassification.DIFFERENT_VARIANT: ("red", "Different product"),
}

_TAGLINE = "Compare prices from trusted retailers. Find the right product, faster."

# Which classifications belong in the primary "Comparable offers" section.
_COMPARABLE_CLASSIFICATIONS = frozenset(
    {
        ShoppingMatchClassification.SAME_PRODUCT,
        ShoppingMatchClassification.SAME_PRODUCT_DIFFERENT_SIZE,
    }
)

# Every card's image gets this width, so images never stretch a card; the
# title is cut to a couple of lines so cards stay a similar height.
_THUMBNAIL_WIDTH = 96
_MAX_TITLE_CHARS = 100

# Characters that Streamlit's markdown would act on (formatting, links, headings,
# and `$...$` math). Listing titles come from the web, so they are escaped.
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+!|~<>$])")


def _escape_markdown(text: str) -> str:
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text)


def _shorten_title(title: str, limit: int = _MAX_TITLE_CHARS) -> str:
    """Collapse whitespace and cut a long listing title at `limit` characters, ending with an ellipsis."""
    title = " ".join(title.split())
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def _display_retailer(offer: ShoppingOffer) -> str:
    """The retailer's plain name ("Amazon", not "amazon.in") when it is a known retailer, else the seller as given."""
    return canonical_retailer(offer.source) or offer.source or "Unknown seller"


def _run_shopping_comparison(product_name: str) -> ShoppingComparisonResult | None:
    """Build the shopping stage, run it, and return the result, or `None` on init/runtime failure."""
    try:
        stage, _storage = build_shopping_stage()
    except (SerpApiSearchError, VisionModelError, ValueError) as exc:
        st.error(f"Could not initialize a required provider: {exc}")
        return None

    with st.spinner("Comparing shopping offers...", show_time=True):
        try:
            return asyncio.run(stage.compare(product_name))
        except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the UI
            st.error(f"Comparison failed: {exc}")
            return None


def _render_shopping_comparison_tab() -> None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        product_name = st.text_input(
            "Product name",
            placeholder="e.g. Tata Salt Iodized Salt 1kg",
            key="shopping_product_name",
            width="stretch",
        )
        compare_clicked = st.button(
            "Compare offers", type="primary", icon=":material/shopping_cart:"
        )

    if compare_clicked:
        cleaned_name = product_name.strip()
        if not cleaned_name:
            st.error("Product name cannot be empty.")
        else:
            st.session_state.pop("last_comparison", None)
            result = _run_shopping_comparison(cleaned_name)
            if result is not None:
                st.session_state["last_comparison"] = result

    if "last_comparison" in st.session_state:
        _render_comparison_result(st.session_state["last_comparison"])


_NO_MATCHING_OFFERS_MESSAGE = "No matching offers found for this product."
_SEARCH_FAILED_MESSAGE = "Couldn't check shopping offers right now. Please try again."


def _group_offers(
    result: ShoppingComparisonResult,
) -> tuple[list[ClassifiedShoppingOffer], list[ClassifiedShoppingOffer], int, int]:
    """
    Split a result into what the consumer sees - `(comparable, related)`,
    already sorted and de-duplicated by `rank_comparison_offers` - and what
    is only counted: `hidden` (different_variant) and `unverified`
    (unclassified). Nothing is removed from `result` itself, so hidden,
    unverified, and duplicate offers stay available for debugging.
    """
    ranked = rank_comparison_offers(result.offers)
    comparable = [c for c in ranked if c.verdict.classification in _COMPARABLE_CLASSIFICATIONS]
    related = [
        c for c in ranked if c.verdict.classification == ShoppingMatchClassification.RELATED_PRODUCT
    ]
    hidden = sum(
        1
        for c in result.offers
        if c.verdict.classification == ShoppingMatchClassification.DIFFERENT_VARIANT
    )
    return comparable, related, hidden, len(result.unclassified_offers)


def _hidden_counts_caption(hidden: int, unverified: int) -> str | None:
    """Compact summary of offers not shown, e.g. "13 unrelated listings hidden · 2 offers couldn't be verified"."""
    parts = []
    if hidden:
        parts.append(f"{hidden} unrelated listing{'' if hidden == 1 else 's'} hidden")
    if unverified:
        parts.append(f"{unverified} offer{'' if unverified == 1 else 's'} couldn't be verified")
    return " · ".join(parts) or None


def _render_comparison_result(result: ShoppingComparisonResult) -> None:
    st.subheader(result.product_name, anchor=False)

    # A failed search must never read as "no matches" - see
    # `ShoppingComparisonResult.search_failed`. Raw errors are logged server-side only.
    if result.search_failed:
        st.warning(_SEARCH_FAILED_MESSAGE)
        return

    comparable, related, hidden, unverified = _group_offers(result)
    counts_caption = _hidden_counts_caption(hidden, unverified)

    if not comparable and not related:
        st.info(_NO_MATCHING_OFFERS_MESSAGE)
        if counts_caption:
            st.caption(counts_caption)
        return

    if comparable:
        st.space("small")
        st.markdown(f"**Comparable offers** · {len(comparable)}")
        for classified in comparable:
            _render_offer_row(
                classified.offer, *_CLASSIFICATION_DISPLAY[classified.verdict.classification]
            )

    if related:
        st.space("medium" if comparable else "small")
        st.markdown(f"**Related alternatives** · {len(related)}")
        st.caption("Not the exact product, but worth considering")
        for classified in related:
            _render_offer_row(
                classified.offer,
                *_CLASSIFICATION_DISPLAY[classified.verdict.classification],
                compact=True,
            )

    if counts_caption:
        st.space("small")
        st.caption(counts_caption)


def _render_offer_row(
    offer: ShoppingOffer, badge_color: _BadgeColor, label: str, *, compact: bool = False
) -> None:
    """
    One offer card: image on the left, title and retailer in the middle, the
    PRICE large on the right with the "View offer" button directly below it.
    `compact` (related alternatives) keeps the same layout with a smaller
    price and a quieter button, so alternatives never compete with the
    comparable offers.
    """
    with st.container(border=True, horizontal=True, vertical_alignment="center", gap="small"):
        st.image(str(offer.thumbnail_url), width=_THUMBNAIL_WIDTH)

        with st.container(width="stretch", gap="xxsmall"):
            st.markdown(f"**{_escape_markdown(_shorten_title(offer.title))}**")
            st.markdown(f":material/storefront: {_escape_markdown(_display_retailer(offer))}")
            st.badge(label, color=badge_color)

        with st.container(width="content", horizontal_alignment="right", gap="xsmall"):
            if not offer.price:
                st.caption("Price unavailable")
            elif compact:
                st.markdown(f"#### {_escape_markdown(offer.price)}")
            else:
                st.subheader(_escape_markdown(offer.price), anchor=False)
            st.link_button(
                "View offer",
                str(offer.product_link),
                icon=":material/open_in_new:",
                type="secondary" if compact else "primary",
            )


def _render_page_header() -> None:
    st.set_page_config(
        page_title="Shop & Compare",
        page_icon=":material/shopping_cart:",
        layout="wide",
    )
    st.title("Shop & Compare", anchor=False)
    st.caption(_TAGLINE)


def main() -> None:
    _render_page_header()

    tab_single, tab_shop = st.tabs(["Single product", "Shop & Compare"])

    with tab_single:
        _render_single_product_tab()

    with tab_shop:
        _render_shopping_comparison_tab()


if __name__ == "__main__":
    main()
