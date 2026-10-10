"""
Offline demo of the Shop & Compare UI on clearly labelled SAMPLE data.

    streamlit run scripts/demo_app.py

Runs the real UI code from `app.py` against a hard-coded result. It needs no
API keys, makes no SerpApi, NVIDIA or retailer requests and uses no API
credits. The retailer names are real approved retailers, but every product,
price and link below is illustrative and does not reflect any live listing.

The product images are placeholder tiles drawn with Pillow and served from a
loopback-only HTTP server, so the demo works without internet access.
"""

from __future__ import annotations

import http.server
import io
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import app  # noqa: E402
from models import (  # noqa: E402
    ClassifiedShoppingOffer,
    ShoppingComparisonResult,
    ShoppingMatchClassification,
    ShoppingMatchVerdict,
    ShoppingOffer,
)

_SAMPLE_BANNER = (
    "**Illustrative sample data.** No live search was run. Products, prices and links below are "
    "made up to show the layout; they are not real listings."
)
_PRODUCT = "Tata Salt Iodized Salt 1kg"
_SAMPLE_LINK = "https://example.com/sample-offer"

_SAME = ShoppingMatchClassification.SAME_PRODUCT
_SIZE = ShoppingMatchClassification.SAME_PRODUCT_DIFFERENT_SIZE
_RELATED = ShoppingMatchClassification.RELATED_PRODUCT
_OTHER = ShoppingMatchClassification.DIFFERENT_VARIANT

# (title, seller as Google Shopping would name it, price, classification, tile colour)
_SAMPLE_OFFERS: list[tuple[str, str, str, ShoppingMatchClassification, tuple[int, int, int]]] = [
    ("Tata Salt Iodized Salt, 1 kg", "BigBasket", "₹28", _SAME, (219, 234, 254)),
    ("Tata Salt Vacuum Evaporated Iodised Salt 1 kg", "Instamart", "₹29", _SAME, (254, 226, 226)),
    ("Tata Salt Iodized Salt 1 kg Pouch", "amazon.in", "₹30", _SAME, (254, 243, 199)),
    ("Tata Salt Iodized Salt (1 kg)", "Flipkart", "₹32", _SAME, (219, 234, 254)),
    ("Tata Salt Iodized Salt 1 kg, Pack of 2", "amazon.in", "₹58", _SIZE, (254, 243, 199)),
    ("Tata Salt Iodized Salt 5 kg", "JioMart Grocery", "₹135", _SIZE, (220, 252, 231)),
    ("Tata Salt Lite Low Sodium Salt 1 kg", "BigBasket", "₹45", _RELATED, (219, 234, 254)),
    ("Tata Rock Salt (Sendha Namak) 1 kg", "blinkit.com", "₹52", _RELATED, (254, 249, 195)),
    ("Tata Salt Shaker 200 g", "Zepto", "₹40", _OTHER, (233, 213, 255)),
    ("Salt Grinder Empty Bottle", "Flipkart", "₹120", _OTHER, (229, 231, 235)),
    ("Himalayan Pink Salt 1 kg", "amazon.in", "₹99", _OTHER, (254, 215, 170)),
]


def _tile(label: str, colour: tuple[int, int, int]) -> bytes:
    """A flat placeholder 'product photo' - deliberately not a real product image."""
    image = Image.new("RGB", (256, 256), colour)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (56, 40, 200, 216), radius=18, fill=(255, 255, 255), outline=(120, 120, 120), width=3
    )
    draw.text((84, 120), "SAMPLE", fill=(90, 90, 90))
    draw.text((84, 138), label[:10], fill=(90, 90, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_resource
def _serve_tiles() -> int:
    """Start (once) a loopback server for the placeholder tiles and return its port."""
    tiles = {
        f"/tile{i}.png": _tile(title, colour)
        for i, (title, _, _, _, colour) in enumerate(_SAMPLE_OFFERS)
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            body = tiles.get(self.path)
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body or b"")))
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return int(server.server_address[1])


def _sample_result() -> ShoppingComparisonResult:
    port = _serve_tiles()
    classified = []
    for index, (title, seller, price, classification, _) in enumerate(_SAMPLE_OFFERS):
        offer = ShoppingOffer(
            position=index + 1,
            title=title,
            source=seller,
            price=price,
            extracted_price=float(price.removeprefix("₹")),
            thumbnail_url=f"http://127.0.0.1:{port}/tile{index}.png",
            product_link=_SAMPLE_LINK,
        )
        verdict = ShoppingMatchVerdict(
            classification=classification, confidence=0.9, reason="sample", model_name="sample"
        )
        classified.append(ClassifiedShoppingOffer(offer=offer, verdict=verdict))
    unverified = [classified[0].offer.model_copy(update={"title": "Sample unverified listing"})]
    return ShoppingComparisonResult(
        product_name=_PRODUCT, offers=classified, unclassified_offers=unverified
    )


def _offline_comparison(product_name: str) -> ShoppingComparisonResult:
    """Stands in for the live search so that pressing "Compare offers" can never make a request."""
    return _sample_result()


def main() -> None:
    app._run_shopping_comparison = _offline_comparison
    app._render_page_header()
    st.info(_SAMPLE_BANNER, icon=":material/science:")

    tab_single, tab_shop = st.tabs(["Single product", "Shop & Compare"])
    with tab_single:
        st.caption(
            "This tab needs API keys and live requests, so it is not part of the offline demo."
        )
    with tab_shop:
        st.session_state.setdefault("last_comparison", _sample_result())
        st.session_state.setdefault("shopping_product_name", _PRODUCT)
        app._render_shopping_comparison_tab()


main()
