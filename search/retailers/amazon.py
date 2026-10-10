"""
Amazon.in product-gallery extractor.

Verified live (see project history): gallery URLs are NOT plain <img src>
attributes. They live inside an inline <script> tag - the `ImageBlockATF`
block - as a JS object literal assigned to a `colorImages` variable. Each
entry has a `"hiRes"` key holding the original, full-resolution image
(e.g. `..._SL1000_.jpg`), distinct from the much smaller `"thumb"`/`"large"`
keys on the same object.

Verified against a real Amazon.in product page: 6 correctly-scoped
`hiRes` images found.

`search_product_page()` drives Amazon's OWN search box (rather than
SerpAPI) to find a candidate product page. Verified live against the raw,
unmodified ERP-style query "FUNFOODS MAYO COMBO 100g" (deliberately
including the internal word "COMBO"): Amazon's search correctly resolved it
to "Dr. Oetker FunFoods Mayonnaise Original Creamy" as the first ORGANIC
result. The first 4 results for that query were sponsored - each result
card (`div[data-component-type="s-search-result"]`) that is sponsored
contains an inner `<span>` whose exact text is "Sponsored"; that is used to
skip sponsored slots rather than guessing a position offset. Also verified:
submitting via the Enter key was unreliable - clicking the actual submit
button (`#nav-search-submit-button`) is what reliably navigates to results.

After the normal search results load, `search_product_page()` ALWAYS
applies Amazon's "Include Out of Stock" refinement before scraping any
result cards, via a URL query parameter - verified live (this session):
the sidebar link labelled "Include Out of Stock" has
`href=".../s?k=<query>&rh=p_n_availability%3A1318485031&..."`, and this
`rh=p_n_availability:1318485031` value was confirmed IDENTICAL across two
unrelated queries ("tomato ketchup" and "jersey kesar flavoured milk"),
so it's a stable, query-independent filter ID safe to append to any
search URL - not query-specific markup that would need re-discovering per
search. Navigating directly to a URL with it appended was confirmed live
to activate the filter (the resulting page then shows a "Remove the
filter Include Out of Stock" link, i.e. the filter registers as applied).
"""

from __future__ import annotations

import logging
import re
import urllib.parse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from search.retailers import GalleryResult, dedupe_preserve_order

logger = logging.getLogger(__name__)

_NAVIGATION_TIMEOUT_MS = 20_000
_HI_RES_PATTERN = re.compile(r'"hiRes":"([^"]+)"')

_COLOR_IMAGES_SCRIPT_JS = """
() => Array.from(document.querySelectorAll('script:not([src])'))
        .map(s => s.textContent)
        .filter(t => t.includes("'colorImages'"))
"""

_HOMEPAGE_URL = "https://www.amazon.in"
_SEARCH_BOX_SELECTOR = "#twotabsearchtextbox"
_SEARCH_SUBMIT_SELECTOR = "#nav-search-submit-button"
_RESULT_CARD_SELECTOR = 'div[data-component-type="s-search-result"]'
_PRODUCT_LINK_SELECTOR = 'a[href*="/dp/"]'
_SPONSORED_LABEL_TEXT = "Sponsored"

# How many non-sponsored candidate URLs to collect, at most - kept in sync
# with `retailer_gallery_stage._MAX_CANDIDATES_PER_RETAILER` (that's the
# actual cap the caller applies; collecting the same number here just
# avoids scanning every remaining result card once enough candidates exist).
_MAX_CANDIDATES = 5

# Amazon's "Include Out of Stock" search refinement - a stable,
# query-independent filter ID, not derived from the search query itself
# (see module docstring for the live verification of both facts).
_OUT_OF_STOCK_FILTER_PARAM = ("rh", "p_n_availability:1318485031")


def _add_out_of_stock_filter(url: str) -> str:
    """Return `url` with Amazon's "Include Out of Stock" refinement appended to its query string."""
    parsed = urllib.parse.urlsplit(url)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_pairs.append(_OUT_OF_STOCK_FILTER_PARAM)
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query_pairs)))


async def extract_gallery(page: Page, product_url: str) -> GalleryResult:
    """Open `product_url` and extract every hi-res gallery image URL."""
    await page.goto(product_url, timeout=_NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded")

    script_texts: list[str] = await page.evaluate(_COLOR_IMAGES_SCRIPT_JS)

    urls: list[str] = []
    for text in script_texts:
        urls.extend(_HI_RES_PATTERN.findall(text))

    return GalleryResult(
        retailer="Amazon",
        product_page_url=product_url,
        image_urls=dedupe_preserve_order(urls),
        source_description=(
            'embedded JSON inside an inline <script> tag '
            '(the "ImageBlockATF" block -> colorImages.initial[].hiRes)'
        ),
    )


async def search_product_page(page: Page, query: str) -> list[str]:
    """
    Search Amazon's own search box for `query` (the caller sends a
    normalized query - see `search.query_normalizer` - packaging/inventory
    codes stripped, known abbreviations expanded; Amazon's own search also
    tolerates a lot of fuzzy/abbreviated naming beyond that, per the module
    docstring) and return up to `_MAX_CANDIDATES` organic (non-sponsored)
    result product URLs, in the order Amazon ranked them, or `[]` if none
    were found. Returning more than one candidate lets the caller try the
    next-ranked result if the top one turns out to be a different product
    (see `retailer_gallery_stage.py`'s multi-candidate loop).

    Amazon.in fronts every request with AWS WAF bot mitigation (verified
    live: the homepage's first response is HTTP 202 with an empty body and
    an `x-amzn-waf-action: challenge` header; the browser's own JS then
    fetches `challenge.js`/`inputs`/`mp_verify` from
    `*.token.awswaf.com` and automatically re-requests the page, which then
    returns a normal HTTP 200 with real content - confirmed live to take
    ~5 seconds end to end). This is NOT new/broken - it's the same class of
    transient challenge already documented for Zepto - and is already
    tolerated here because `.click()`'s built-in actionability wait (up to
    `_NAVIGATION_TIMEOUT_MS`) polls for `_SEARCH_BOX_SELECTOR` rather than
    checking once immediately, so it naturally waits out the challenge. The
    logging below exists so that if this ever fails again, the logs show
    exactly which step/selector didn't resolve in time rather than a bare
    `None`.
    """
    response = await page.goto(_HOMEPAGE_URL, timeout=_NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded")
    logger.debug(
        "[Amazon] homepage navigation: status=%s url=%s", response.status if response else None, page.url
    )

    search_box = page.locator(_SEARCH_BOX_SELECTOR)
    try:
        await search_box.click(timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        box_count = await page.locator(_SEARCH_BOX_SELECTOR).count()
        body_sample = (await page.evaluate("() => document.body.innerText"))[:300]
        logger.warning(
            "[Amazon] search box selector=%r never became clickable (matched %d element(s)); "
            "page.url=%s title=%r body_sample=%r - likely an unresolved AWS WAF challenge or a "
            "changed homepage DOM",
            _SEARCH_BOX_SELECTOR,
            box_count,
            page.url,
            await page.title(),
            body_sample,
        )
        return []
    await search_box.fill(query)
    logger.debug("[Amazon] query filled: %r", query)

    # Clicking the real submit button, not pressing Enter: verified live
    # that Enter alone does not reliably submit this search form.
    try:
        await page.locator(_SEARCH_SUBMIT_SELECTOR).click(timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        logger.warning(
            "[Amazon] search submit button selector=%r never became clickable; page.url=%s",
            _SEARCH_SUBMIT_SELECTOR,
            page.url,
        )
        return []
    logger.debug("[Amazon] search submitted; page.url=%s", page.url)

    try:
        await page.wait_for_selector(_RESULT_CARD_SELECTOR, timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        body_sample = (await page.evaluate("() => document.body.innerText"))[:300]
        logger.warning(
            "[Amazon] no result cards (selector=%r) appeared after submitting search; "
            "page.url=%s title=%r body_sample=%r",
            _RESULT_CARD_SELECTOR,
            page.url,
            await page.title(),
            body_sample,
        )
        return []

    # Always include out-of-stock results before scraping any result card -
    # see module docstring for the live verification of this URL parameter.
    # If applying it fails for any reason (network hiccup, Amazon changing
    # this refinement ID), fall back to the normal (unfiltered) results
    # already loaded above rather than failing the whole search.
    try:
        await page.goto(
            _add_out_of_stock_filter(page.url), timeout=_NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded"
        )
        await page.wait_for_selector(_RESULT_CARD_SELECTOR, timeout=_NAVIGATION_TIMEOUT_MS)
        logger.debug("[Amazon] out-of-stock filter applied; page.url=%s", page.url)
    except PlaywrightTimeoutError:
        logger.debug(
            "[Amazon] out-of-stock filter navigation/re-wait timed out - continuing with the "
            "results already loaded before this step, per the fall-through-on-failure design"
        )

    result_cards = page.locator(_RESULT_CARD_SELECTOR)
    count = await result_cards.count()
    logger.info("[Amazon] %d result card(s) found", count)

    sponsored_count = 0
    candidates: list[str] = []
    for index in range(count):
        if len(candidates) >= _MAX_CANDIDATES:
            break

        card = result_cards.nth(index)
        is_sponsored = await card.get_by_text(_SPONSORED_LABEL_TEXT, exact=True).count() > 0
        if is_sponsored:
            sponsored_count += 1
            continue

        link = card.locator(_PRODUCT_LINK_SELECTOR).first
        href = await link.get_attribute("href")
        if href:
            # `get_attribute` returns the raw, unresolved attribute -
            # verified live that Amazon's result links are relative paths.
            resolved = href if href.startswith("http") else f"{_HOMEPAGE_URL}{href}"
            candidates.append(resolved)

    deduped = dedupe_preserve_order(candidates)
    logger.info(
        "[Amazon] %d candidate(s) collected (skipped %d sponsored card(s) across %d card(s) scanned): %r",
        len(deduped),
        sponsored_count,
        count,
        deduped,
    )
    return deduped
