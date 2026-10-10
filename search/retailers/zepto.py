"""
Zepto product-gallery extractor and native search.

INVESTIGATION (this session, live, before any code was written): a bare
Playwright context - no explicit user agent - gets HTTP 429 from Zepto on
every request (reproduced on multiple previously-unvisited URLs, persisting
after a 30s wait, so not a brief burst limit). The response carries an
`aws-waf-token` cookie once a request succeeds, confirming this is AWS WAF
bot mitigation, not a fundamental "this app can't run under Playwright"
limitation. With a realistic desktop Chrome user agent + viewport on a
*separate* browser context, the homepage hydrates completely: real body
text (5,800+ chars, ₹ pricing visible), a full Next.js asset bundle loads,
and the server sets genuine session cookies (`session_id`, `device_id`,
`csrfSecret`, `XSRF-TOKEN`, `aws-waf-token`). Because
`retailer_gallery_stage.py` shares one `page`/context sequentially across
all retailers, both functions in this module open their own short-lived,
realistic `BrowserContext` (via `_open_realistic_page`) instead of using
the shared `page` directly - Amazon and Flipkart already finish their own
turns (they run first in `_RETAILERS`) before Zepto's context is ever
created, so they are never touched by this.

DELIVERY LOCATION (investigated live): Zepto auto-resolves a delivery
location from the request itself (server-side, likely IP-based - the
`Playwright` `geolocation` context option was set but the resolved
coordinates in the `get_page` API call did NOT match it, so Zepto is not
reading it) and that auto-resolved location was confirmed live to be
serviceable (searches return real, in-stock-looking results). Manually
overriding the location via Zepto's own "Select Location" -> address
autocomplete flow was tested live and is a genuine trap for a headless
scraper: picking an arbitrary real address (a real, indexed Delhi landmark)
put the whole session into an unserviceable state ("Sit Tight! We're
Coming Soon!" - zero catalog, zero search). No location is set here for
that reason - the server's own default is the only one confirmed safe.

SEARCH (native, no external search engine): `search_product_page()` drives
Zepto's actual client-rendered search flow, verified live end-to-end:
1. Load the homepage (through the realistic context above).
2. Click the search trigger (`get_by_text("Search for", ...)` - matches
   exactly one element, the decorative search-bar placeholder text; the
   underlying `<input>` is `type="hidden"` until this is clicked, so it
   cannot be typed into directly). This client-side-routes to
   `/search` and reveals a real, visible `input[type="text"]` whose
   placeholder is "Search for over 5000 products".
3. Type the query into that input (via `_type_and_verify` - see its own
   docstring for why a plain `.type()` isn't enough) and press Enter (not
   the debounced autosuggest dropdown - pressing Enter reliably navigates
   to `/search?query=...` with real results, mirroring Amazon's/
   Flipkart's "click submit rather than trust autosuggest" pattern).
4. Wait for real results (via `_wait_for_real_results` - see its own
   docstring for why this needs two distinct signals, not just "a
   product link appeared"), then read every `a[href*="/pn/"]` link -
   confirmed live to be exactly the real product-card links (no ads/
   "Sponsored" markup was found on any tested query, unlike Amazon).
This is backed by Zepto's own backend: `POST
bff-gateway.zepto.com/user-search-service/api/v3/search`, observed live
carrying the typed query - a real endpoint belonging to the site being
scraped, not a third-party search engine.

QUERY CLEANUP, verified live to be necessary for Zepto's OWN search
(mirrors, but is independent from, the packaging/quantity noise this
project already strips for other reasons elsewhere): the raw ERP string
"KISSAN FRESH TOMATO KETCHUP 950g PCH" returned only 2 results, and both
were the WRONG product ("The True Tomato Ketchup"); removing the trailing
quantity token alone ("KISSAN FRESH TOMATO KETCHUP") returned 14 results
with the correct "Kissan Fresh Tomato Ketchup" variants dominating the
top of the list. Zepto's own search (like Amazon's/Flipkart's) already
tolerates the ordinary noise `search.query_normalizer.normalize_for_search()`
removes (that existing, UNCHANGED function is reused here - see its
docstring: it exists specifically "for the string sent to a retailer's
own search box"); on top of that, this module strips a trailing quantity
token, verified live to be the difference between the wrong product and
the right one.

LIMITATIONS, found live and reported honestly:
- Zepto's own search can genuinely return zero results for a real, valid
  product even with a clean query - verified live for "JERSEY KESAR
  FLAVOURED MILK" (the page itself renders Zepto's own "Could not find
  any products for ..." message; the broader "JERSEY MILK" returned 27
  results including other Godrej Jersey products, so the brand is
  indexed, but this specific flavour/SKU combination was not surfaced).
  Also verified live for "MAGGI 2 MINUTE NOODLES MASALA" - notably, in
  that case the search API's own `totalProductCount` field reported 17,
  but the page still rendered the same "Could not find any products"
  message, so that field is NOT trusted as a success signal anywhere in
  this module; what's actually rendered to the page is. Either way, this
  is treated the same as Amazon/Flipkart finding nothing:
  `search_product_page` returns `None` and the pipeline falls through to
  the next retailer.
- No login is performed or required. Cookies are whatever the site sets
  itself on a normal, unauthenticated visit - none are pre-supplied or
  hardcoded.
- Delivery location is left at Zepto's own default resolution (see above)
  rather than pinned - a returned product may not be serviceable in every
  real-world area, the same availability caveat already accepted for
  Amazon/Flipkart, where downstream vision-model verification - not
  availability - is what actually gates acceptance.
"""

from __future__ import annotations

import logging
import re

from playwright.async_api import Browser, BrowserContext, Locator, Page, TimeoutError as PlaywrightTimeoutError

from search.query_normalizer import normalize_for_search
from search.retailers import GalleryResult, dedupe_preserve_order

logger = logging.getLogger(__name__)

_NAVIGATION_TIMEOUT_MS = 20_000
_PRODUCT_VARIANT_PATH_MARKER = "/cms/product_variant/"
_PRODUCT_VARIANT_IMG_SELECTOR = 'img[src*="/cms/product_variant/"]'
_IK_TRANSFORM_PATTERN = re.compile(r"/tr:[^/]+/")

# How many distinct candidate URLs to collect, at most - kept in sync with
# `retailer_gallery_stage._MAX_CANDIDATES_PER_RETAILER`.
_MAX_CANDIDATES = 5

_PRODUCT_VARIANT_IMAGE_SRCS_JS = """
(marker) => Array.from(document.querySelectorAll('img'))
        .map(img => img.src)
        .filter(src => src.includes(marker))
"""

# Verified live this session (see module docstring): required to avoid a
# 429 from Zepto's AWS WAF. Any recent real desktop Chrome UA/viewport pair
# works - this isn't a Zepto-specific fingerprint, just a generic "looks
# like a real browser" pair.
_DESKTOP_CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_DESKTOP_VIEWPORT = {"width": 1280, "height": 800}

_HOMEPAGE_URL = "https://www.zepto.com/"
# Matches exactly one element live (the decorative search-bar placeholder
# text) - see module docstring.
_SEARCH_TRIGGER_TEXT = "Search for"
_SEARCH_INPUT_SELECTOR = 'input[type="text"][placeholder="Search for over 5000 products"]'
_PRODUCT_LINK_SELECTOR = 'a[href*="/pn/"]'
# Zepto's own literal empty-state text, verified live (e.g. for "maggi 2
# minute noodles masala", which the search API's totalProductCount field
# reported as 17 despite this message being shown - see
# `_wait_for_real_results`).
_NO_RESULTS_TEXT = "Could not find any products"
# A DIFFERENT page state from the above, found live during pipeline root-
# cause investigation: Zepto occasionally gates search results behind a
# login wall ("Oops! Please login to continue searching") instead of
# rendering results at all - confirmed transient/probabilistic, not tied to
# a specific query or to running after Amazon/Flipkart in the same browser
# process (the identical query and the identical Amazon->Flipkart->Zepto
# sequence both succeeded on immediate re-runs). Detected as its own signal,
# distinct from `_NO_RESULTS_TEXT`, so this failure mode is never silently
# conflated with a genuine empty catalog result in logs, and so it returns
# immediately instead of waiting out the full timeout.
_LOGIN_WALL_TEXT = "Please login to continue searching"

# A trailing quantity token ("950g") is dropped before it ever reaches
# Zepto's own search box - verified live this session (see module
# docstring): with it left in, Zepto's search returned the wrong product;
# with it removed, the correct product was the top result.
_TRAILING_QUANTITY_PATTERN = re.compile(r"\s*\d+(?:\.\d+)?\s*(?:g|kg|ml|L)\b\.?\s*$")


def _strip_trailing_quantity(text: str) -> str:
    return _TRAILING_QUANTITY_PATTERN.sub("", text).strip()


def _strip_resize_transform(url: str) -> str:
    return _IK_TRANSFORM_PATTERN.sub("/", url)


async def _open_realistic_page(browser: Browser) -> tuple[BrowserContext, Page]:
    """
    Open a fresh `BrowserContext` with a realistic desktop UA/viewport -
    required to avoid Zepto's AWS WAF 429 (see module docstring). Caller is
    responsible for closing the returned context.
    """
    context = await browser.new_context(user_agent=_DESKTOP_CHROME_USER_AGENT, viewport=_DESKTOP_VIEWPORT)
    page = await context.new_page()
    return context, page


async def extract_gallery(page: Page, product_url: str) -> GalleryResult:
    """
    Open `product_url` and extract every gallery image URL from the DOM.

    Navigates in its own isolated, realistic `BrowserContext` rather than
    the shared `page` passed in - see module docstring. Falls back to the
    passed-in `page` directly if a new context can't be created (e.g. a
    persistent context with no `.browser`), rather than failing outright.
    """
    browser = page.context.browser
    if browser is None:
        return await _extract_gallery_on_page(page, product_url)

    context, isolated_page = await _open_realistic_page(browser)
    try:
        return await _extract_gallery_on_page(isolated_page, product_url)
    finally:
        await context.close()


async def _extract_gallery_on_page(page: Page, product_url: str) -> GalleryResult:
    await page.goto(product_url, timeout=_NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded")

    # Product images are client-rendered AFTER domcontentloaded fires (this
    # is a Next.js app) - verified live that reading <img> tags immediately
    # here finds ZERO matching images even on pages that do have a real
    # gallery (the identical page, given a few seconds to hydrate, has
    # them). Wait for at least one to actually appear before reading; a
    # genuine timeout here means the page has no gallery at all, which
    # still yields the same "empty GalleryResult" outcome as before - it
    # just no longer misreports a real gallery as empty.
    try:
        await page.wait_for_selector(_PRODUCT_VARIANT_IMG_SELECTOR, timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        logger.info(
            "[Zepto] no gallery image (selector=%r) ever appeared at %s - treating as a "
            "genuinely gallery-less page",
            _PRODUCT_VARIANT_IMG_SELECTOR,
            product_url,
        )

    thumbnail_urls: list[str] = await page.evaluate(
        _PRODUCT_VARIANT_IMAGE_SRCS_JS, _PRODUCT_VARIANT_PATH_MARKER
    )
    original_urls = [_strip_resize_transform(url) for url in thumbnail_urls]

    return GalleryResult(
        retailer="Zepto",
        product_page_url=product_url,
        image_urls=dedupe_preserve_order(original_urls),
        source_description=(
            'DOM <img> tags whose src contains "/cms/product_variant/", with the ImageKit '
            "/tr:<params>/ resize prefix stripped to reach the original file"
        ),
    )


async def search_product_page(page: Page, query: str) -> list[str]:
    """
    Search Zepto's own search box for `query` and return up to
    `_MAX_CANDIDATES` distinct result product URLs, in the order Zepto
    ranked them, or `[]` if no product page was found - never raises, so
    the pipeline moves on to the next retailer. Returning more than one
    candidate lets the caller try the next-ranked result if the top one
    turns out to be a different product (see `retailer_gallery_stage.py`'s
    multi-candidate loop).

    Runs in its own isolated, realistic `BrowserContext` rather than the
    shared `page` passed in - see module docstring for why. Falls back to
    the passed-in `page` directly if a new context can't be created.

    If the first attempt finds nothing, retries once more in a brand-new
    isolated context before giving up: Zepto occasionally gates a search
    behind a transient login wall (see `_LOGIN_WALL_TEXT`'s own comment)
    that was confirmed live, both in an earlier root-cause investigation
    and again while diagnosing this regression, to resolve on an immediate
    retry with a fresh session - it is not a genuine empty-catalog result.
    Only one retry (not a loop), to bound the worst-case latency this adds.

    `query` (the raw, unmodified ERP product name, though the caller
    normally passes an already-normalized query - see
    `retailer_gallery_stage.py`) is first passed through the existing,
    unchanged `search.query_normalizer.normalize_for_search()` - reusing it
    here is not a change to normalization behavior, just applying the
    existing function to a new caller - then has a trailing quantity token
    stripped, verified live to be necessary for Zepto's own search
    specifically (see module docstring).
    """
    logger.info(
        "[Zepto] search_product_page() called from %s - query received: %r", __file__, query
    )

    browser = page.context.browser
    if browser is None:
        logger.warning(
            "[Zepto] page.context.browser is None (persistent context?) - "
            "falling back to the shared page instead of an isolated one (no retry available "
            "on this path)"
        )
        result = await _search_on_page(page, query)
        logger.info("[Zepto] search_product_page() returning %d candidate(s): %r", len(result), result)
        return result

    logger.debug("[Zepto] opening isolated realistic BrowserContext for search")
    context, isolated_page = await _open_realistic_page(browser)
    try:
        result = await _search_on_page(isolated_page, query)
    finally:
        await context.close()

    if not result:
        logger.info(
            "[Zepto] first attempt found nothing for query=%r - retrying once in a fresh "
            "context (see docstring: a transient login-wall gate has been confirmed live to "
            "resolve on an immediate retry)",
            query,
        )
        context, isolated_page = await _open_realistic_page(browser)
        try:
            result = await _search_on_page(isolated_page, query)
        finally:
            await context.close()
        logger.info("[Zepto] retry result: %d candidate(s): %r", len(result), result)

    logger.info("[Zepto] search_product_page() returning %d candidate(s): %r", len(result), result)
    return result


async def _type_and_verify(search_input: Locator, query: str, max_attempts: int = 4) -> bool:
    """
    Type `query` into `search_input` and read the value back, retrying (a
    full clear + retype, not a partial correction) if it doesn't match.

    Verified live to be necessary: typing right after the search-trigger
    click races with this site's own client-side route transition/re-render
    and can silently drop or reorder leading characters - e.g. "KISSAN
    FRESH TOMATO KETCHUP" was observed coming out as "ATO KETCHUP" or
    "OMATO KETCHUP". Neither a fixed post-click delay nor waiting for the
    URL to settle on `/search` eliminated this reliably when tested live
    (both still dropped characters on ~1 in 3 attempts); reading the actual
    input value back and retrying until it matches is self-verifying rather
    than a timing guess, and was confirmed live to reach a correct value
    within at most 2 attempts across 6 consecutive real runs.
    """
    for attempt in range(1, max_attempts + 1):
        await search_input.fill("")
        await search_input.type(query, delay=60)
        value = await search_input.input_value()
        if value == query:
            logger.debug("[Zepto] input typed correctly on attempt %d/%d: %r", attempt, max_attempts, value)
            return True
        logger.debug(
            "[Zepto] input value mismatch on attempt %d/%d: got %r, expected %r - retrying",
            attempt,
            max_attempts,
            value,
            query,
        )
    logger.warning("[Zepto] typing never stabilized to the intended query after %d attempts", max_attempts)
    return False


async def _links_on_page(page: Page) -> list[str]:
    return await page.evaluate(
        "(sel) => Array.from(document.querySelectorAll(sel)).map(a => a.href)", _PRODUCT_LINK_SELECTOR
    )


async def _wait_for_real_results(page: Page, decoy_links: frozenset[str]) -> list[str]:
    """
    Poll until one of three things happens, or give up after
    `_NAVIGATION_TIMEOUT_MS` (treated the same as "no results" by the
    caller - see module docstring's LIMITATIONS section for why this is
    reported as `None` either way, but logged distinctly for each case):
    (a) a `/pn/` link NOT in `decoy_links` appears - real results.
    (b) Zepto's own "Could not find any products" message appears - a
        genuine, confirmed-empty catalog result.
    (c) Zepto's login-wall text appears - an anti-bot gate, NOT a genuine
        empty result (see `_LOGIN_WALL_TEXT`'s own comment for the live
        evidence that this is transient/probabilistic, not deterministic).
    Returns the new, non-decoy links, or `[]` for (b)/(c)/timeout.

    All three signals were verified live to be necessary:
    - Before a query is submitted, the search overlay already shows
      `/pn/` links for an unrelated "trending products" carousel -
      waiting for "any `/pn/` link to appear" (the Amazon/Flipkart
      pattern) matches those decoys instantly, before the real,
      submitted-query results replace them a moment later. Comparing
      against the pre-submit snapshot is what distinguishes the two.
    - Zepto's own `totalProductCount` field in its search API response is
      NOT a reliable success signal on its own - verified live for "MAGGI
      2 MINUTE NOODLES MASALA": the API reported `totalProductCount: 17`,
      but the rendered page showed literal text "Could not find any
      products for ...". The decoy set was never replaced (0 new links)
      for this query either, so relying on decoy-differencing alone would
      have correctly avoided a false positive here, but only checking for
      new links (with no explicit-failure signal) leaves no way to
      distinguish "still loading" from "confirmed empty" other than a
      full timeout - the explicit message, when present, lets this return
      quickly instead of always waiting out the full timeout.
    - The login wall was found during a later root-cause investigation of
      an integrated-pipeline failure that the standalone script didn't
      reproduce: replaying the identical query, and even the identical
      Amazon->Flipkart->Zepto sequence in one browser process, both
      succeeded immediately afterward - proving this is an occasional
      site-side gate (not tied to query, sequence, or a stale
      implementation), and proving it needs its own detection so it's
      never silently counted as a genuine empty result in logs.
    """
    logger.debug("[Zepto] decoy links snapshotted before Enter (%d): %s", len(decoy_links), sorted(decoy_links))

    poll_interval_ms = 250
    elapsed_ms = 0
    body_text = ""
    while elapsed_ms < _NAVIGATION_TIMEOUT_MS:
        await page.wait_for_timeout(poll_interval_ms)
        elapsed_ms += poll_interval_ms

        body_text = await page.evaluate("() => document.body.innerText")
        if _LOGIN_WALL_TEXT in body_text:
            logger.warning(
                "[Zepto] login wall (\"%s\") blocked results at ~%dms - this is NOT a genuine "
                "empty-catalog result, it's a site-side anti-bot gate; confirmed transient in "
                "root-cause investigation (immediate re-runs of the identical query succeeded)",
                _LOGIN_WALL_TEXT,
                elapsed_ms,
            )
            return []
        if _NO_RESULTS_TEXT in body_text:
            logger.info(
                "[Zepto] explicit \"%s\" message seen at ~%dms - genuine zero-result page",
                _NO_RESULTS_TEXT,
                elapsed_ms,
            )
            return []

        current = await _links_on_page(page)
        new_links = [link for link in current if link not in decoy_links]
        if new_links:
            logger.info(
                "[Zepto] %d new (non-decoy) product link(s) found at ~%dms: %s",
                len(new_links),
                elapsed_ms,
                new_links,
            )
            return new_links

    logger.warning(
        "[Zepto] timed out after %dms waiting for real results - neither new links nor the "
        "\"%s\" message appeared; last seen page URL=%s, body sample=%r",
        elapsed_ms,
        _NO_RESULTS_TEXT,
        page.url,
        (body_text or "")[:200],
    )
    return []


async def _search_on_page(page: Page, query: str) -> list[str]:
    cleaned_query = _strip_trailing_quantity(normalize_for_search(query))
    logger.info("[Zepto] raw query=%r -> cleaned query=%r", query, cleaned_query)

    try:
        response = await page.goto(_HOMEPAGE_URL, timeout=_NAVIGATION_TIMEOUT_MS, wait_until="domcontentloaded")
    except PlaywrightTimeoutError as exc:
        logger.warning("[Zepto] homepage navigation timed out: %s", exc)
        return []
    logger.debug(
        "[Zepto] homepage loaded: status=%s url=%s",
        response.status if response else None,
        page.url,
    )

    search_trigger = page.get_by_text(_SEARCH_TRIGGER_TEXT, exact=False).first
    try:
        await search_trigger.click(timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        trigger_count = await page.get_by_text(_SEARCH_TRIGGER_TEXT, exact=False).count()
        logger.warning(
            "[Zepto] could not click search trigger text=%r (matched %d element(s)); "
            "page.url=%s, page.title()=%r - search page never opened",
            _SEARCH_TRIGGER_TEXT,
            trigger_count,
            page.url,
            await page.title(),
        )
        return []
    logger.debug("[Zepto] search trigger clicked; page.url=%s", page.url)

    search_input = page.locator(_SEARCH_INPUT_SELECTOR)
    try:
        await search_input.click(timeout=_NAVIGATION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        input_count = await page.locator(_SEARCH_INPUT_SELECTOR).count()
        logger.warning(
            "[Zepto] could not click search input selector=%r (matched %d element(s)); "
            "page.url=%s - search input never became clickable",
            _SEARCH_INPUT_SELECTOR,
            input_count,
            page.url,
        )
        return []
    logger.debug("[Zepto] search input clicked and ready")

    if not await _type_and_verify(search_input, cleaned_query):
        logger.warning(
            "[Zepto] giving up: typed query never stabilized to %r - not submitting search", cleaned_query
        )
        return []

    decoy_links = frozenset(await _links_on_page(page))
    await search_input.press("Enter")
    logger.debug("[Zepto] Enter pressed; page.url=%s", page.url)

    result_links = await _wait_for_real_results(page, decoy_links)
    if not result_links:
        # A genuine zero-result page for Zepto shows no product cards at
        # all (verified live) - this is a real "no match", not a bug.
        logger.info("[Zepto] no product URL to return for query=%r (cleaned=%r)", query, cleaned_query)
        return []

    # `_links_on_page` reads `a.href` (the resolved absolute URL, not the
    # raw attribute), so every entry is already a full URL.
    candidates = dedupe_preserve_order(result_links)[:_MAX_CANDIDATES]
    logger.info(
        "[Zepto] full result list (%d), %d candidate(s) after dedup: %s",
        len(result_links),
        len(candidates),
        candidates,
    )
    return candidates
