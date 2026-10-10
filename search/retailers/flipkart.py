"""
Flipkart product-gallery extractor.

Verified live (see project history): naive DOM scanning (all <img> tags on
the page, or climbing up from the main product image to a shared ancestor
<div>) picks up UNRELATED images from a nearby "similar products"
recommendation carousel that reuses the exact same CSS classes and
image-size bucket as the real product gallery - there is no reliable
class-name-based way to tell them apart. The reliable source instead is
Flipkart's own JSON-LD structured data:
`<script id="jsonLD" type="application/ld+json">`, whose `image` array is
correctly scoped to the current product AND gives 1500x1500 URLs (higher
resolution than the same images' <img> tags elsewhere on the page, which
serve 800x1070).

IMPORTANT CAVEAT, found while verifying live: Flipkart's own JSON-LD can
itself contain mismatched images due to real marketplace catalog
data-quality issues - one live product's `image` array included an
unrelated tea product photo alongside its own correct images. This is a
genuine data-quality issue on Flipkart's side, not a scraping bug - it
means the caller's product-match verification step is load-bearing for
Flipkart specifically, not just a formality.

`search_product_page()` drives Flipkart's OWN search box (rather than
SerpAPI) to find a candidate product page. Verified live against the raw,
unmodified ERP-style query "FUNFOODS MAYO COMBO 100g": the first result
was "funfoods-dr-oetker-veg-mayonnaise-original-pizza-topping" - a correct
match despite the messy ERP wording. No sponsored/ad markup was found on
that results page (no element with the exact text "Ad" or similar); if
Flipkart does inject ads for other queries, that has not been observed and
this does not attempt to filter for it - unlike Amazon, where sponsored
slots were directly observed and are explicitly skipped. Also verified:
submitting via the Enter key was unreliable here too - clicking the actual
submit button is what reliably navigates to results. The search input is
matched by its placeholder text and the submit button by
`button[type="submit"]`, not by its CSS class, since Flipkart's classes are
build-hashed and change across deploys (the same reason `flipkart.py`
avoids class-based selectors for gallery extraction above).

Unlike Amazon, Flipkart does NOT currently expose an "include out of
stock"/"show unavailable" filter via a URL parameter (or at all) -
verified live across two unrelated categories ("tomato ketchup" and
"laptop"): the sidebar's only "Availability" filter option is "GST
Invoice Available" in both cases, with no stock-status option anywhere on
the page. So `search_product_page()` still ALWAYS attempts to find and
click such a filter via Playwright first (a real, scoped search - not a
hardcoded no-op - see `_apply_out_of_stock_filter_if_present`), but today
that search finds nothing and the function falls through to scraping the
normal results, exactly as this project's spec requires when a retailer
doesn't expose the filter. If Flipkart ever adds one, this starts applying
it automatically without needing a code change.
"""

from __future__ import annotations

import json
import re

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from search.retailers import GalleryResult, dedupe_preserve_order

_NAVIGATION_TIMEOUT_MS = 20_000

# Broad, case-insensitive text match for a possible "include out of stock" /
# "show unavailable" filter control - not currently found anywhere on
# Flipkart's search results (see module docstring), so this is intentionally
# a wide net rather than a specific selector nobody has ever seen render.
_OUT_OF_STOCK_FILTER_TEXT_PATTERN = re.compile(r"out of stock|unavailable", re.IGNORECASE)

_JSON_LD_SCRIPT_JS = """
() => { const el = document.getElementById('jsonLD'); return el ? el.textContent : null; }
"""

_HOMEPAGE_URL = "https://www.flipkart.com"
# The plain placeholder selector matches TWO inputs on this page - a real,
# editable one and a `readonly` duplicate (verified live, via a Playwright
# "strict mode violation" error listing both) - `:not([readonly])`
# disambiguates down to the one that can actually be typed into.
_SEARCH_BOX_SELECTOR = 'input[placeholder="Search for Products, Brands and More"]:not([readonly])'
_SEARCH_SUBMIT_SELECTOR = 'button[type="submit"]'
_PRODUCT_LINK_SELECTOR = 'a[href*="/p/"]'

# How many distinct candidate URLs to collect, at most - kept in sync with
# `retailer_gallery_stage._MAX_CANDIDATES_PER_RETAILER`. Each product card
# on a real results page was found (live) to contain 3 separate anchors to
# the identical URL (image link, title link, wrapping link), so candidates
# must be deduped before slicing to this count - see `dedupe_preserve_order`
# below.
_MAX_CANDIDATES = 5


async def extract_gallery(page: Page, product_url: str) -> GalleryResult:
    """Open `product_url` and extract every gallery image URL from its JSON-LD."""
    await page.goto(product_url, timeout=_NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded")

    raw: str | None = await page.evaluate(_JSON_LD_SCRIPT_JS)

    urls: list[str] = []
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            items = data if isinstance(data, list) else [data]
            for item in items:
                images = item.get("image") if isinstance(item, dict) else None
                if isinstance(images, list):
                    urls.extend(images)
                elif isinstance(images, str):
                    urls.append(images)

    return GalleryResult(
        retailer="Flipkart",
        product_page_url=product_url,
        image_urls=dedupe_preserve_order(urls),
        source_description=(
            'JSON-LD structured data (<script id="jsonLD" type="application/ld+json"> -> "image" array); '
            "NOTE: can include mismatched images due to real Flipkart catalog data-quality issues, "
            "not a scraping bug - product-match verification is load-bearing here"
        ),
    )


_LOGIN_POPUP_SELECTOR = "div.mcO4kT.RFBkxv"


async def _dismiss_login_popup(page: Page) -> None:
    """
    Flipkart shows a "Login / Get access to your Orders, Wishlist and
    Recommendations" popup shortly after the homepage loads, which
    intercepts clicks on the search box underneath it (verified live: a
    Playwright click on the search input failed with the popup's container
    reported as intercepting pointer events). Pressing Escape reliably
    dismisses it - verified live - without depending on the popup's
    build-hashed CSS classes.
    """
    try:
        await page.wait_for_selector(_LOGIN_POPUP_SELECTOR, timeout=3_000)
    except PlaywrightTimeoutError:
        return
    await page.keyboard.press("Escape")
    try:
        await page.wait_for_selector(_LOGIN_POPUP_SELECTOR, state="detached", timeout=3_000)
    except PlaywrightTimeoutError:
        pass


async def _apply_out_of_stock_filter_if_present(page: Page) -> None:
    """
    Best-effort: look for a sidebar filter control whose text suggests
    "include/show out-of-stock or unavailable products" and click it if
    one exists, then wait for the results to refresh. Excludes matches
    that are inside a product link (`a[href*="/p/"]`) so an individual
    product's own "Out of Stock" badge is never mistaken for a search-wide
    filter control. If nothing matches (the current reality on Flipkart -
    see module docstring) or a click doesn't visibly do anything, this
    returns quietly and the caller proceeds with the normal results -
    never raises, per this stage's "don't fail the search over a filter
    that doesn't exist" requirement.
    """
    try:
        candidates = page.get_by_text(_OUT_OF_STOCK_FILTER_TEXT_PATTERN)
        count = await candidates.count()
    except Exception:  # noqa: BLE001 - best-effort only
        return

    for index in range(count):
        candidate = candidates.nth(index)
        try:
            inside_product_link = await candidate.locator(
                "xpath=ancestor-or-self::a[contains(@href, '/p/')]"
            ).count()
        except Exception:  # noqa: BLE001 - best-effort only
            inside_product_link = 0
        if inside_product_link:
            continue

        try:
            await candidate.click(timeout=3_000)
            await page.wait_for_load_state("networkidle", timeout=5_000)
            return
        except Exception:  # noqa: BLE001 - try the next match, or give up quietly
            continue


async def search_product_page(page: Page, query: str) -> list[str]:
    """
    Search Flipkart's own search box for `query` (the caller sends a
    normalized query - see `search.query_normalizer` - packaging/inventory
    codes stripped, known abbreviations expanded; Flipkart's own search
    also handles fuzzy/abbreviated naming beyond that, per the module
    docstring) and return up to `_MAX_CANDIDATES` distinct result product
    URLs, in the order Flipkart ranked them, or `[]` if none were found.
    Returning more than one candidate lets the caller try the next-ranked
    result if the top one turns out to be a different product (see
    `retailer_gallery_stage.py`'s multi-candidate loop).
    """
    await page.goto(_HOMEPAGE_URL, timeout=_NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded")
    await _dismiss_login_popup(page)

    search_box = page.locator(_SEARCH_BOX_SELECTOR)
    await search_box.click(timeout=_NAVIGATION_TIMEOUT_MS)
    await search_box.fill(query)
    # Clicking the real submit button, not pressing Enter: verified live
    # that Enter alone does not reliably submit this search form.
    await page.locator(_SEARCH_SUBMIT_SELECTOR).click(timeout=_NAVIGATION_TIMEOUT_MS)

    try:
        await page.wait_for_selector(_PRODUCT_LINK_SELECTOR, timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        return []

    # Always attempt to include out-of-stock results before scraping any
    # product link - see module docstring: Flipkart doesn't currently
    # expose this as a filter at all, so this normally finds nothing and
    # falls through to the results already loaded above.
    await _apply_out_of_stock_filter_if_present(page)
    try:
        await page.wait_for_selector(_PRODUCT_LINK_SELECTOR, timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        return []

    hrefs = await page.locator(_PRODUCT_LINK_SELECTOR).evaluate_all(
        "(els) => els.map(el => el.getAttribute('href'))"
    )
    resolved = [href if href.startswith("http") else f"{_HOMEPAGE_URL}{href}" for href in hrefs if href]
    return dedupe_preserve_order(resolved)[:_MAX_CANDIDATES]
