> **Archived engineering log.** This is the original chronological handoff, written before the
> project became Shop & Compare. It describes behavior that no longer exists (Playwright retailer
> scrapers, Flipkart/Zepto search, the Batch (Excel) tab, ...) and is kept only for the rationale
> behind past decisions. For the current architecture see [`HANDOFF.md`](../../HANDOFF.md).

# Project Handoff — Product Image Retriever

> Generated as a complete state snapshot so work can continue in a new chat
> without losing context. Paste this whole document into the new chat.

---

# Project Overview

**Goal**: Given a product name (typically an **ERP/internal inventory name**,
e.g. `"KISSAN FRESH TOMATO KETCHUP 950g PCH"` or `"FUNFOODS MAYO COMBO
100g"`), automatically retrieve real product images from the internet,
verify they actually depict the requested product (tolerating retailer
naming differences), and save them locally in a predictable folder/file
layout.

**Current architecture** (as of this handoff): a **retailer-first**
pipeline, on **Amazon, Flipkart, and Zepto** (Blinkit and BigBasket remain
removed entirely - see "Retailer simplification" below; Zepto was removed
in that same simplification but **re-added this session** as a genuinely
first-class retailer - see "Zepto re-added as a retailer..." in Problems
Already Solved). **IMPORTANT - reverted back to the ORIGINAL
single-candidate selection logic** (see "Full revert of candidate
ranking/filtering/brand" at the end of Problems Already Solved): a later
session tried adding brand-aware query broadening, a 5-candidate pool with
ranking, and cheap title-based pre-filtering, but the user found it caused
REAL products to start returning WRONG matches on some items, and asked
for a full revert of all of that - keeping only the retailer removal, the
Amazon→Flipkart→Google order, and the NVIDIA retry logic. **Do not
re-introduce brand-aware queries, candidate pooling/ranking, or
title-based filtering without the user explicitly asking again** - it was
tried and explicitly rolled back for a real accuracy regression, not just
a preference change. Re-adding Zepto did not touch any of this - it's a
new retailer entry, not a change to candidate selection.

The current (reverted, working) flow:
1. Searches Amazon, then Flipkart, then Zepto, using each retailer's OWN
   native search box - no brand combination, no query broadening, no
   candidate pooling, takes ONLY the first likely result from each.
   Amazon and Flipkart use the **raw, unmodified ERP product name**
   as-is. Zepto's own search needed light query cleanup to work reliably
   (see item 13 in Problems Already Solved) - the existing, unchanged
   `normalize_for_search()` plus a Zepto-local trailing-quantity strip -
   verified live that Zepto's own search engine, unlike Amazon's/
   Flipkart's, returns the WRONG product for a raw ERP string with a
   trailing quantity/packaging code still attached.
2. Opens that page and extracts its **entire image gallery** exactly as
   the retailer shows it (no front/back/side classification).
3. Sends only the **first/primary gallery image** to a Vision-Language
   Model (Nemotron 3 Nano Omni, via NVIDIA Build - **migrated from Qwen
   this session, after a first attempt at Mistral Medium 3.5 failed to
   ever respond**, see items 17-18 in Problems Already Solved) together
   with the **original, unmodified ERP product name**, asking "is this
   the same commercial product?" (tolerant of naming/wording differences,
   strict about genuine variant/flavor/type differences).
   `verify_product_match` retries transient NVIDIA Build failures
   (HTTP 500s, malformed/missing/empty responses, timeouts) a small
   number of times before giving up - this retry logic is the one piece
   of the later optimization work that WAS kept (see
   `classifier/nemotron.py`).
4. If accepted: downloads and saves the **whole gallery**, in original
   order, as `image_01.jpg, image_02.jpg, ...` plus a `metadata.json`. Stop.
5. If rejected: try the next retailer in priority order (Amazon → Flipkart).
6. If **both retailers reject**: fall back to the **original Google
   Images pipeline** (still fully functional, not deleted) — which
   downloads 5 candidates per side (front/back) via SerpAPI Google Images,
   sends all 5 to the vision model in one comparative-ranking request per
   side, and saves `front.<ext>` / `back.<ext>` + `metadata.json`.

**High-level workflow** (text form):
```
ERP product name
   -> RetailerGalleryStage.run(product_name)
        search_query_text = normalize_for_search(product_name)  # logged only; see below
        for retailer in [Amazon, Flipkart]:
            product_url = await retailer["search_fn"](page, product_name)  # RAW name, both retailers native
            extract full gallery from product_url
            download ONLY first gallery image
            ask vision model: verify_product_match(first_image, ORIGINAL product_name)  # retries transient failures
            if accepted (is_match AND confidence >= 0.6):
                download REST of gallery
                persist_gallery() -> image_01.ext, image_02.ext, ...
                write metadata.json
                RETURN result
            else:
                continue to next retailer
        # both retailers rejected/failed:
        fall back to ProductImageSelectionStage.run(product_name) (Google Images path)
   -> ProductImageResult
```
(`search_query_text` is computed and logged for parity with the original
design, but - since Zepto/Blinkit's SerpAPI-based path is gone and both
remaining retailers always use their own native search with the RAW
name - it is not actually applied to any search call today.)

Everything is built with **dependency injection** and **clean
architecture** principles: interfaces (`interfaces.py`) define abstract
contracts; concrete adapters implement them; orchestrators (`selection_stage.py`,
`retailer_gallery_stage.py`) wire injected dependencies together and contain
no provider-specific logic themselves.

**Important running theme all session**: "do not guess selectors" — every
retailer-specific CSS selector, JSON path, or DOM structure claim in this
codebase was derived by actually opening a live page (via an interactive
browser tool) and inspecting it first, with real evidence (byte comparisons,
live counts, etc.) cited in code comments/docstrings, not assumed.

---

# Folder Structure

```
product_image_retriever root (= this directory)/
├── run.py                         # ACTIVE CLI entry point; also exports build_stage()/describe_result() for reuse
├── app.py                         # ACTIVE Streamlit UI entry point (thin frontend over run.build_stage())
├── config.py                      # Settings (pydantic-settings), env var loading
├── models.py                      # All Pydantic domain models
├── interfaces.py                  # Abstract base classes (ports)
├── prompts.py                     # All LLM prompt templates
├── verification_overrides.py      # Deterministic post-verification brand-mismatch guardrail (see its own docstring)
├── selection_stage.py             # Google-Images-based workflow; the FALLBACK stage
├── retailer_gallery_stage.py      # PRIMARY orchestrator (retailer-first workflow)
├── HANDOFF.md                     # This document
│
├── batch/                         # Excel batch-processing (app.py's "Batch (Excel)" tab) - reuses
│   │                               # RetailerGalleryStage/build_stage() unchanged, once per Excel row
│   ├── column_detection.py        # detect_columns() (required EAN/Description) + detect_brand_column()
│   │                               # (optional Brand column, used only for verification_overrides.py)
│   ├── excel_import.py            # read_product_rows() - .xlsx -> list[ProductRow] via openpyxl
│   ├── processor.py                # run_batch() - async generator; sequential (no concurrency),
│   │                               # calls the SAME stage.run(description, erp_brand=...) per row,
│   │                               # yields BatchRowResult; only zips a row's folder if it actually
│   │                               # has images (a never-created folder for a zero-result product
│   │                               # would otherwise crash the whole batch)
│   ├── packaging.py                # zip_product_directory() / write_results_workbook() /
│   │                               # build_final_bundle() - zips already-persisted output, no new
│   │                               # storage logic
│   └── models.py                   # ProductRow (incl. optional brand), BatchRowResult dataclasses
│
├── search/
│   ├── query_generator.py         # SimpleQueryGenerator - builds "<name> front image"/"back image" (used by selection_stage.py fallback)
│   ├── query_normalizer.py        # normalize_for_search() - strips ERP packaging/inventory codes
│   │                               # (PP, PCH, PET, BTL, ...); sent to ALL THREE retailers' own search
│   │                               # boxes by retailer_gallery_stage.py - verification always still
│   │                               # gets the original, unmodified ERP name
│   ├── serpapi.py                 # SerpApiImageSearchProvider - SerpAPI "engine=google_images" - used by selection_stage.py fallback
│   └── retailers/
│       ├── __init__.py            # GalleryResult dataclass, dedupe_preserve_order()
│       ├── amazon.py              # search_product_page() (native, returns up to 5 ranked candidates) + extract_gallery()
│       ├── flipkart.py            # search_product_page() (native, returns up to 5 ranked candidates) + extract_gallery()
│       └── zepto.py               # search_product_page() (native, with a one-retry-on-empty-result
│                                   # safety net) + extract_gallery() (waits for client-rendered images
│                                   # before reading the DOM)
│
├── classifier/
│   └── nemotron.py                # NemotronVisionModel - THE vision model client (NVIDIA Build / Nemotron 3 Nano Omni)
│
├── download/
│   └── downloader.py               # HttpImageDownloader - the ONLY downloader, fully generic/reused everywhere
│
├── storage/
│   └── local_filesystem.py        # LocalFilesystemStorage - the ONLY storage backend
│
├── utils/
│   ├── slugify.py                 # slugify() - product-name -> filesystem-safe folder name
│   ├── hashing.py                 # sha256_hex() - used by downloader for checksums
│   └── logging_setup.py           # configure_logging()
│
├── tests/
│   └── test_models.py             # 6 tests, all passing - only covers models.py, NOT retailer stage/normalizer/etc.
│
├── images/                         # Single root: images/<product_slug>/... final results ONLY.
│                                   # No downloads/ or output/ subfolder - candidate downloads use
│                                   # ephemeral OS temp directories instead.
│
└── .env                            # Real API keys (NVIDIA_API_KEY, PIR_SERPAPI_API_KEY) - see Environment Variables section
```

**Key files and their single responsibility:**

| File | Responsibility |
|---|---|
| `retailer_gallery_stage.py` | **PRIMARY orchestrator.** Retailer-first search (top-5 candidates/retailer) → extract → verify → deterministic brand-mismatch override → full-gallery download, with fallback to Google Images (`ENABLE_GOOGLE_FALLBACK` - a one-line switch to temporarily isolate the retailer-only path for testing). |
| `verification_overrides.py` | Deterministic, prompt-independent guardrail applied AFTER the vision model's verdict: force-rejects a candidate when the ERP's own brand and the model's extracted brand are both confidently known and clearly different, regardless of the model's own `is_match`. One-directional (never flips a reject into an accept). |
| `selection_stage.py` | **FALLBACK orchestrator.** Google Images (via SerpAPI) → download 5/side → vision-model comparative ranking → save front/back. |
| `classifier/nemotron.py` | All Nemotron/NVIDIA Build API interaction. Three independent capabilities (see below). |
| `search/retailers/*.py` | One file per retailer: native search (returns ranked candidate URLs) and gallery extraction. |
| `search/query_normalizer.py` | Cleans ERP names for ALL THREE retailers' own search boxes (packaging/inventory code removal, abbreviation expansion) - never used for verification. |
| `download/downloader.py` | The one and only image downloader, used by both orchestrators identically. |
| `storage/local_filesystem.py` | The one and only storage backend; has both `persist_image`/`persist_result` (front/back mode) and `persist_gallery` (gallery mode). |
| `models.py` | All Pydantic data models shared across the whole system. |
| `interfaces.py` | Abstract ports (`ImageSearchProvider`, `ImageDownloader`, `VisionModel`, `StorageProvider`, `QueryGenerator`). Concrete classes implement these but new capabilities (e.g. `verify_product_match`, `select_best_candidate`, `persist_gallery`) are added as **concrete-only** methods, NOT added to these interfaces, to avoid forcing unrelated adapters to support them. |
| `config.py` | `pydantic-settings`-based `Settings` class; reads `.env`. |
| `run.py` | CLI entry point; wires everything together (`build_stage()`), reused by `app.py` and `batch/processor.py`. |

---

# Current Pipeline

Full execution flow from entering a product name to images being saved on
disk, exactly as `run.py` drives it today:

1. **`run.py:main()`** calls `configure_logging("INFO")`, then
   `prompt_for_product_name()` (loops until non-empty input).
2. **`run.py:build_stage()`** constructs:
   - `HttpImageDownloader()` (no config needed)
   - `NemotronVisionModel()` (reads `NVIDIA_API_KEY` from `config.Settings.nvidia_api_key`)
   - `LocalFilesystemStorage()` (reads `config.Settings.images_dir`)
   - `ProductImageSelectionStage(...)` as the **fallback_stage**, itself wired
     with `SerpApiImageSearchProvider()` (reads `PIR_SERPAPI_API_KEY`), the
     same downloader/vision_model/storage instances above.
   - `RetailerGalleryStage(...)` as the **primary stage**, wired with the
     same downloader/vision_model/storage instances, plus the fallback_stage.
3. **`RetailerGalleryStage.run(product_name)`**:
   a. Computes `search_query_text = normalize_for_search(product_name)`
      once (only actually used later for Zepto/Blinkit).
   b. Launches one headless Chromium browser via Playwright, opens one page,
      reuses it across all 4 retailer attempts.
   c. For each retailer in `_RETAILERS` order (Amazon, Flipkart, Zepto,
      Blinkit), calls `_try_retailer(page, retailer, product_name,
      search_query_text)`:
      - **Amazon/Flipkart** (have `search_fn`): calls
        `retailer["search_fn"](page, product_name)` — i.e.
        `amazon.search_product_page` / `flipkart.search_product_page` —
        using the **ORIGINAL** ERP name typed directly into that retailer's
        own search box.
      - **Zepto/Blinkit** (no `search_fn`): calls
        `find_retailer_product_page(search_query_text, serpapi_key, domain,
        path_markers)` — SerpAPI `engine=google` + `site:<domain>` — using
        the **NORMALIZED** query.
      - If a product URL was found: calls `retailer["extractor"](page,
        product_url)` (`amazon.extract_gallery`, `flipkart.extract_gallery`,
        `zepto.extract_gallery`, or `blinkit.extract_gallery`) to get a
        `GalleryResult` (full list of image URLs in original order + a
        human-readable `source_description` of where they came from).
      - If the gallery is non-empty: downloads ONLY `image_urls[0]` via
        `self._downloader.download(candidate, destination_dir)` →
        `DownloadedImage`. Destination dir = a fresh, private OS temp
        directory (`tempfile.TemporaryDirectory()`) created per retailer
        attempt — **not** anywhere under `images/`. This directory (and
        whatever's in it) is always removed when the attempt's `with` block
        exits, whether accepted or rejected.
      - Calls `self._vision_model.verify_product_match(first_downloaded.local_path,
        product_name)` — **always the ORIGINAL product_name**, never the
        normalized query — → `ProductMatchVerdict`.
      - If `verdict.is_match` is `True` AND `verdict.confidence >= 0.6`
        (`_DEFAULT_MIN_MATCH_CONFIDENCE`): **accepted**. Downloads every
        remaining gallery image concurrently (`asyncio.gather`, skip
        individual failures), combines `[first_downloaded, *rest]`, calls
        `self._persist(...)`, and **returns** the result — no further
        retailers are tried.
      - Otherwise (rejected, or any step raised): logs why, returns `None`,
        loop continues to the next retailer.
   d. If no retailer returned a result: logs and calls
      `self._fallback_stage.run(product_name)` (the Google Images pipeline,
      described below), returning whatever it produces.
4. **`_persist(...)`** (inside `RetailerGalleryStage`): calls
   `self._storage.persist_gallery(downloaded_images, product_name)` → copies
   every downloaded image, in order, straight from the temp directory to
   `<images_dir>/<slugify(product_name)>/image_01.<ext>, image_02.<ext>,
   ...` (the temp directory is removed right after, by the enclosing
   `with tempfile.TemporaryDirectory()` block). Builds a `metadata` dict (see Current Storage Layout /
   Verification Logic below) and a `ProductImageResult`
   (`front_image_path`/`back_image_path`/`front_confidence`/`back_confidence`
   all left `None` — not meaningful in gallery mode). Calls
   `self._storage.persist_result(result)` → writes `metadata.json` in the
   same output folder.
5. **Fallback path** (`ProductImageSelectionStage.run(product_name)`, only
   reached if all 4 retailers rejected): for `side` in `["front", "back"]`
   independently — search `SimpleQueryGenerator` builds `"<name> front
   image"` / `"<name> back image"` via `SerpApiImageSearchProvider`
   (`engine=google_images`) → download top 5 candidates into
   `<per-run OS temp dir>/front|back/` (never under `images/`; removed once
   both sides are persisted) → send **all 5 in one request** to
   `NemotronVisionModel.select_best_candidate(image_paths, product_name, side)`
   → get back the winning index + a `ClassificationVerdict` → if its
   confidence clears the (currently unused/0.0-default) floor, persist via
   `LocalFilesystemStorage.persist_image(...)` as `front.<ext>` /
   `back.<ext>`, build a `ProductImageResult` with real
   `front_confidence`/`back_confidence`, and `persist_result()` →
   `metadata.json`. Front and back pools are NEVER merged/compared against
   each other.
6. **`run.py:print_result()`**: checks `result.metadata.get("retailer")` —
   if present, prints the retailer-gallery summary (retailer name, URL,
   gallery size, each saved path); if absent, prints the front/back summary.
   Always prints the `metadata.json` path (via
   `storage.product_dir(product_name)`).

---

# Components Implemented

- **Amazon provider** (`search/retailers/amazon.py`) — ✅ **Finished, fully
  verified live** (both DOM inspection AND actual Playwright execution from
  this sandbox). `extract_gallery()`: parses an inline `<script>` tag (the
  `ImageBlockATF` block) for a `colorImages` JS object; regexes out every
  `"hiRes":"..."` URL (original, full-resolution, e.g. `..._SL1000_.jpg`).
  `search_product_page()`: drives `#twotabsearchtextbox` +
  `#nav-search-submit-button` (Enter key alone does NOT reliably submit —
  verified), filters out sponsored results (any result card containing a
  `<span>` with exact text `"Sponsored"`), returns first organic
  `a[href*="/dp/"]` (resolved to absolute URL — raw `href` attribute is
  relative). A later session briefly changed this to
  `search_product_candidates()` (returning up to 5 titled candidates for
  ranking/filtering) but that was fully reverted at the user's request
  after a real accuracy regression - see Problems Already Solved.

- **Flipkart provider** (`search/retailers/flipkart.py`) — ✅ **Finished,
  fully verified live.** `extract_gallery()`: parses `<script id="jsonLD"
  type="application/ld+json">`'s `image` array (1500×1500 originals).
  **Known caveat** (found live, not a bug): this JSON-LD can itself contain
  a mismatched image due to real Flipkart catalog data-quality issues (one
  product's array included an unrelated tea photo) — verification is
  load-bearing here. `search_product_page()`: drives
  `input[placeholder="Search for Products, Brands and More"]:not([readonly])`
  (plain placeholder selector matches 2 inputs, one is a readonly duplicate
  — verified via a Playwright strict-mode error) + `button[type="submit"]`.
  **Must dismiss a "Login" popup first** (`_dismiss_login_popup`) — it
  intercepts clicks on the search box otherwise; dismissed via pressing
  `Escape` (verified live — does NOT depend on Flipkart's build-hashed CSS
  classes). Returns first `a[href*="/p/"]`, resolved to absolute. A later
  session briefly changed this to a title-aware `search_product_candidates()`
  (using `a[href*="/p/"][title]` to skip an image-wrapper anchor bug found
  live) but that was fully reverted at the user's request after a real
  accuracy regression elsewhere in the same change - see Problems Already
  Solved. The underlying wrapper-anchor issue this fix addressed is real
  (confirmed live for "KISSAN FRESH TOMATO KETCHUP 950g PCH": the first
  `a[href*="/p/"]` pointed to an unrelated "Little Joys" product) but is
  currently NOT specifically guarded against, per the user's explicit
  instruction to restore the original implementation exactly.

- **Zepto provider** (`search/retailers/zepto.py`) — ⚠️ **`extract_gallery()`
  finished and verified** (via interactive browser tool: `<img>` tags whose
  `src` contains `/cms/product_variant/`, strip the ImageKit
  `/tr:<params>/` resize prefix to get the original — verified live via
  byte-size comparison, 29KB rendered vs 107KB original). **Search is
  SerpAPI-based, NOT native** — Zepto's own homepage/search renders **0
  characters of content** in this sandbox (bot-detection or geolocation
  gate that never resolves), so native search was never attempted/verified
  for Zepto. Direct product-page navigation (what SerpAPI naturally
  provides) works. **Caveat**: even with a known-good product URL, a bare
  Playwright process from this sandbox gets a fully empty response (no
  title, no body text) — worked fine via the interactive browser tool.
  Likely environment-specific (network/fingerprint), not necessarily true
  on a residential network.

- **Blinkit provider** (`search/retailers/blinkit.py`) — ⚠️ **Same
  situation as Zepto.** `extract_gallery()` finished and verified:
  `<img class*="ProductCarousel__CarouselImage">`, strip the Cloudflare
  `/cdn-cgi/image/<params>/` resize prefix (verified: ~480×480 resized vs
  96KB original). **Search is SerpAPI-based** — Blinkit's homepage triggers
  a location-picker whose "enter address manually" flow depends on Google
  Places Autocomplete, which never populated suggestions in testing. A bare
  Playwright process from this sandbox gets an **explicit Cloudflare
  "access denied"** block even hitting a known-good product URL directly —
  worked fine via the interactive browser tool. Same environment-specific
  caveat as Zepto.

- **Vision-model verification** (`classifier/nemotron.py`, migrated from
  Qwen this session via a failed Mistral attempt - see items 17-18 in
  Problems Already Solved) — ✅ **Finished, three independent, coexisting
  capabilities** on `NemotronVisionModel`, and (unlike the earlier Mistral
  attempt) fully live-verified end-to-end:
  1. `classify_image()` / `classify()` — judges ONE image in isolation
     (front/back/other/unknown + is_correct_product + confidence). Still
     present, satisfies `interfaces.VisionModel`, but **not used by either
     active orchestrator today** (superseded by `select_best_candidate` for
     the fallback stage).
  2. `select_best_candidate()` — comparative ranking: given up to 5 image
     paths + product_name + side, sends ALL in ONE request, model picks the
     single best. Used by `selection_stage.py` (fallback stage). 5-image
     support confirmed live for Nemotron (see item 18).
  3. `verify_product_match()` — **THE one used by the primary retailer
     stage.** One image + original ERP product_name → `ProductMatchVerdict`
     (is_match, confidence, reason, brand, product_name, weight). Uses
     `PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT` — redesigned once already
     (see Problems Already Solved) to be tolerant of ERP-vs-retailer naming
     differences; unchanged by the Nemotron migration.
  All three share the same low-level request machinery
  (`_classify()` → `_post_chat_completion()`), differing only in
  message-builder and response-parser. **Confirmed working end-to-end
  against the real NVIDIA Build API** - a real full retailer-pipeline run
  (Amazon) accepted and persisted a real product's gallery, and Flipkart
  and Zepto were separately confirmed too (see item 18 for the full
  verification writeup).

- **Downloader** (`download/downloader.py`, `HttpImageDownloader`) — ✅
  **Finished.** Implements `interfaces.ImageDownloader`. `download()`
  (singular, `ImageCandidate` → `DownloadedImage`) is used by both
  orchestrators. `download_images()` (batch, `list[str]` → `list[Path]`,
  skip-on-failure) exists but note **`retailer_gallery_stage.py` does NOT
  use this batch method** — it uses `download()` in a loop via
  `asyncio.gather` instead, to preserve full `DownloadedImage` provenance
  (source_provider, checksum, etc.) per image, which the bare-`Path`
  batch method would discard. Validates images via Pillow
  (`Image.verify()`), computes SHA-256 checksums, handles
  `InvalidImageUrlError` / `ImageNetworkError` / `ImageTimeoutError` /
  `CorruptImageError`.

- **Storage** (`storage/local_filesystem.py`, `LocalFilesystemStorage`) — ✅
  **Finished.** Implements `interfaces.StorageProvider`
  (`persist_image`/`persist_result`, used by the fallback stage) PLUS a
  non-interface `persist_gallery()` method (used by the primary retailer
  stage) that saves an ordered list as `image_01.<ext>, image_02.<ext>,
  ...`. Both write `metadata.json` via the same `persist_result()`.

- **Retailer gallery stage** (`retailer_gallery_stage.py`,
  `RetailerGalleryStage`) — ✅ **Finished for Amazon/Flipkart, partially
  environment-limited for Zepto/Blinkit** (their extraction logic is done
  and verified; their SEARCH step works via SerpAPI but hasn't been
  attempted as native search). Depends concretely on `NemotronVisionModel`
  and `LocalFilesystemStorage` (not the generic interfaces), by design.

- **Local storage / output layout** — ✅ Finished, see Current Storage
  Layout section.

- **UI status** — ✅ **Streamlit UI added (`app.py`)**, in addition to the
  original CLI (`run.py`). Thin frontend only, per explicit instruction: it
  contains no retrieval/verification/search/download/storage logic. It
  reuses `run.build_stage()` (composition root, unchanged) to construct the
  same `RetailerGalleryStage` + `ProductImageSelectionStage` fallback, runs
  `stage.run(product_name)` via `asyncio.run(...)` exactly like `run.py`
  does, and reuses a new `run.describe_result()` helper (extracted from
  `run.py:print_result()` in the same change) to decide what to display
  without re-implementing that interpretation in `app.py`. Progress is shown
  by mirroring the pipeline's existing `logger.info(...)` calls into a
  Streamlit placeholder via one extra `logging.Handler` — no new logging
  calls were added to any business module. Run with `streamlit run app.py`.
  `streamlit>=1.35` was added to `pyproject.toml`'s dependencies. Verified
  live end-to-end in-browser this session (real SerpAPI/NVIDIA calls):
  Amazon and Flipkart correctly rejected, Zepto/Blinkit found no gallery
  (per their known sandbox limitation), fell back to Google Images, and the
  UI correctly rendered "Retailer used: none (Google Images fallback)",
  both images, and the full `metadata.json`.

- **Excel batch processing added** (`batch/` package + `app.py`'s new
  "Batch (Excel)" tab, this session): lets a user upload an arbitrary ERP
  Excel export (`.xlsx`) and runs the SAME `RetailerGalleryStage` (via
  `run.build_stage()`, unchanged) once per row, sequentially (no
  concurrency, by explicit requirement). Column layout is never assumed -
  `batch/column_detection.py` matches normalized header text against known
  aliases to find the EAN and Product Description columns anywhere in the
  header row; every other column is ignored. Only the Product Description
  is used to search (never the EAN) - the EAN is used solely to name each
  product's output ZIP (`<EAN>.zip`, contents = whatever's already in that
  product's `images/<slug>/` folder: `image_01.jpg...`/`metadata.json` or
  `front.jpg`/`back.jpg`/`metadata.json`, unchanged). Rows missing EAN or
  Product Description are skipped (not stopped) and recorded in
  `results.xlsx` with a reason; an exception on one row logs a warning and
  continues to the next row. After all rows: `results.xlsx` (EAN, Product
  Description, Status, Source Used, Number of Images, Notes) plus every
  per-product ZIP are bundled into one final ZIP, offered via
  `st.download_button`. All packaging uses ephemeral OS temp directories
  (`tempfile.mkdtemp()`, `shutil.rmtree()` after reading the final bytes
  into memory) - nothing batch-related is written under `images/` beyond
  what `RetailerGalleryStage`/`ProductImageSelectionStage` already persist
  per product. `openpyxl>=3.1` was added to `pyproject.toml` (new
  dependency; reads/writes `.xlsx` directly, no `pandas` dependency added).
  Verified this session: offline checks (column detection across the
  README's example ERP layouts, blank-row skipping, float-EAN-to-clean-
  string conversion, ZIP structure, results.xlsx shape) plus one real,
  live 3-row run through the actual batch code (via a harness bypassing
  only the literal native file-picker click, which the browser tool here
  can't drive) - one row correctly ran the real pipeline end-to-end
  (Amazon accepted, 8 images persisted and zipped), and the other two were
  correctly skipped with the right reasons ("Missing EAN" /
  "Missing product description"), all reflected correctly in the live
  per-row progress feed, the results table, and the download button.

- **Selector (old per-image confidence selector)**
  (`selector/confidence_selector.py`, `ConfidenceBasedSelector`) — 🗑️
  **Legacy/unused.** Was the original selection mechanism before the
  comparative-ranking redesign. Left in the codebase (not deleted) per an
  earlier explicit instruction to "keep architecture" / not delete things
  unnecessarily. Not imported by either active orchestrator today.

- **`search/google_images.py` / `search/google_search.py`** — 🗑️
  **Legacy/unused Playwright scrapers.** Predate the SerpAPI-based
  approach. Not wired into any active code path. `search/google_images.py`
  in particular has a long debugging history (see Problems Already Solved)
  but is no longer what actually runs — the active fallback stage uses
  `search/serpapi.py`'s `SerpApiImageSearchProvider` (SerpAPI
  `engine=google_images`) instead.

- **`pipeline.py` / `main.py`** — 🗑️ **Legacy/unused/never finished.** An
  earlier, more generic DI-based pipeline concept. `main.py:build_pipeline()`
  still raises `NotImplementedError`. `run.py` is the real entry point and
  does not go through these at all.

---

# Important Design Decisions

- **Why retailer-first instead of Google Images**: retailer product pages
  give exact, high-resolution, unwatermarked gallery images unambiguously
  tied to one specific SKU. Generic image search (even via SerpAPI) returns
  a mixed bag of quality/relevance. Google Images is NOT removed — it's
  kept as the fallback when retailer-first entirely fails, since it still
  reliably finds *something* even for products with no clean retailer
  listing.

- **Why Amazon/Flipkart now use native search**: live-verified that both
  sites' own search engines correctly resolve raw, unmodified ERP-style
  names (e.g. "FUNFOODS MAYO COMBO 100g" → correct product, despite
  "COMBO" being pure internal jargon) as well as or better than Google. No
  evidence of bot-detection blockers on their search-results pages
  specifically (as opposed to their homepages' other features). The
  trade-off (discussed explicitly with the user before implementing) is
  that interactive search-box automation is inherently a more
  bot-detection-visible pattern than a single deep-linked page load — this
  was accepted as a reasonable risk for these two specific sites, but was
  explicitly NOT extended to Zepto/Blinkit given direct evidence their
  search/homepage flows are currently blocked in this environment.

- **[OUTDATED for Zepto - see item 13 in Problems Already Solved; kept
  for Blinkit history] Why Zepto/Blinkit used SerpAPI**: both sites' own
  search/homepage flows were evidenced (in an earlier session) to be
  blocked or non-functional (Zepto: fully empty page; Blinkit:
  Google-Places-Autocomplete-dependent location gate that never resolved).
  For Zepto specifically, a LATER session re-investigated with a realistic
  browser context (UA + viewport) instead of a bare default one, and found
  the earlier "blocked" finding was actually AWS WAF bot mitigation, not a
  fundamental limitation - Zepto's own native search now works and SerpAPI
  was fully removed from `zepto.py`. Blinkit remains removed and untested
  against this fix (it was never re-investigated).

- **Why only the first product result is opened** (not multiple
  candidates per retailer): keeps the search step fast and simple.
  Correctness is NOT the search step's job — it's enforced entirely by the
  verification step (the vision model - Qwen originally, migrated to
  Nemotron 3 Nano Omni this session, see items 17-18). If the first result is
  wrong, verification rejects it and the system moves to the next
  RETAILER (not the next result on the same retailer). This was an
  explicit simplification choice in the user's stated workflow, not
  something separately re-litigated.

- **Why only the first gallery image is sent to the vision model for
  verification**: cost/latency — verifying an entire gallery per retailer
  attempt would be expensive and slow, especially across multiple
  retailer attempts. A single primary/hero product photo is normally
  sufficient to judge brand/product-family/variant/weight/packaging. The
  one observed limitation (see Known Bugs) is that a single photo
  sometimes doesn't visually show a printed weight; the verifier prompt
  was explicitly rewritten (see Problems Already Solved) so this is
  treated as unknown, not a mismatch, and does not by itself cause a
  rejection.

- **Why the full gallery is downloaded only AFTER verification accepts**:
  avoids wasting downloads/bandwidth on retailers that get rejected. Only
  one image (the first) is downloaded speculatively per retailer attempt;
  the rest are only fetched once that retailer has been confirmed correct.

- **Why Google Images was NOT removed, only demoted to fallback**: it's a
  reliable last-resort for products with no clean retailer presence
  (discontinued products, very niche items, or retailers this project
  doesn't yet support). Removing it entirely would reduce overall recall
  for no benefit.

- **Why the vision model is the source of truth**: search (whether native
  or SerpAPI-driven) can and does return wrong candidates — directly
  observed twice in an earlier session (Amazon returning a *different*
  Kohinoor rice variant as the top organic result; Flipkart's own JSON-LD
  structured data containing an unrelated tea product image due to a real
  catalog data-quality issue on Flipkart's side). The architecture treats
  search results as **candidates only** — the vision model's verification
  (Qwen originally, Nemotron 3 Nano Omni from this session - see items
  17-18) is the single, final gate that decides acceptance, regardless of
  how confident the search step "felt."

- **Why `ProductMatchVerdict` is a separate model from
  `ClassificationVerdict`**: different responsibility. `ClassificationVerdict`
  judges an image's `side` (front/back/other) for a candidate already
  presumed roughly correct. `ProductMatchVerdict` judges the prior question
  — "is this even the right product at all?" — which has no notion of
  `side`. Reusing one model for both would force a meaningless `side` value
  onto product-match results.

- **Why the query normalizer is rule-based, not an LLM call**: the hard
  requirement "never invent a different product" is best guaranteed
  structurally. A fixed lookup-table substitution (expand known
  abbreviations, remove known noise words, canonicalize units) mechanically
  cannot add information that wasn't in the input. An LLM rewrite, even
  carefully instructed, could still occasionally embellish/guess — exactly
  the failure mode being avoided. This normalizer is ONLY used for the
  Zepto/Blinkit SerpAPI search path; Amazon/Flipkart's native search
  receives the raw, unmodified ERP name directly (their own search engines
  handle fuzzy naming better than our own crude normalization would).

- **Why `classify_image`/`select_best_candidate`/`verify_product_match`
  all coexist on `NemotronVisionModel` without removing earlier ones**:
  established pattern all session — new capabilities are added as new,
  concrete-only methods, never by modifying or removing what already
  works, so nothing that depends on an earlier method silently breaks.

- **Why new capabilities (`verify_product_match`, `select_best_candidate`,
  `persist_gallery`) are NOT added to `interfaces.py`**: they're specific
  to one concrete adapter each (the vision model's comparative/verification abilities,
  local-filesystem's gallery-saving ability) and forcing them onto the
  abstract interface would require every hypothetical alternative
  provider/backend to support them, which isn't justified yet with only
  one real implementation of each interface.

- **Storage architecture was simplified to a single root (this session)**:
  the old two-directory layout (`images/downloads/` scratch candidates +
  `images/output/` final results) was replaced with one root, `images/`,
  containing only final per-product folders (`images/<product_slug>/...`).
  `config.Settings` no longer has `downloads_dir`/`output_dir` fields — only
  `images_dir`. Both stages now download their *unaccepted* candidates
  (the retailer stage's single verification probe per retailer;
  the fallback stage's 5-per-side comparison pool) into a private,
  per-run/per-attempt **OS temp directory** (`tempfile.TemporaryDirectory()`
  / `tempfile.mkdtemp()`), never under `images/`. Only images that are
  actually accepted/selected are copied into
  `images/<product_slug>/...` — the temp directory is always removed
  afterward, win or lose. This was an **explicit trade-off the user chose**
  (via `AskUserQuestion`) over keeping a persistent `images/downloads/`
  scratch tree: `_cleanup_temp_files` (the old manual per-file delete helper
  in `selection_stage.py`) and `_resolve_downloads_dir` (in both stages)
  were removed as obsolete — cleanup is now just letting the temp-directory
  context manager exit.

- **"Development mode: downloaded candidates are NOT deleted" is NO LONGER
  true** as of this session's storage cleanup — it was explicitly superseded
  by the decision above. Rejected/losing candidate images are no longer kept
  on disk for inspection; only the final accepted/selected images survive,
  under `images/<product_slug>/...`. If earlier context (or memory) says
  otherwise, this note is the more recent, authoritative one.

---

# Current Provider Order

**Retailer priority** (`retailer_gallery_stage.py:_RETAILERS`), confirmed
explicitly with the user. Blinkit and BigBasket remain removed entirely
(see "Retailer simplification..." in Problems Already Solved). **Zepto was
added back this session** (see item 13, "Zepto's own native search..." in
Problems Already Solved - item 12's SerpAPI-based first attempt was
explicitly rejected by the user and fully replaced) as a genuinely
first-class retailer, not a special case:

1. **Amazon** (`amazon.in`) — native search
2. **Flipkart** (`flipkart.com`) — native search
3. **Zepto** (`zepto.com`) — Zepto's OWN native search UI, driven via
   Playwright exactly like Amazon/Flipkart (no SerpAPI, no Google, no
   external search engine of any kind). Requires a realistic desktop UA +
   viewport on its own isolated `BrowserContext` - a bare default
   Playwright context gets HTTP 429 from Zepto's AWS WAF. Gallery
   extraction is native Playwright DOM scraping, same as Amazon/Flipkart,
   in that same isolated context.

If all three reject, produce no viable candidates, or fail → fall back to
`ProductImageSelectionStage` (Google Images via SerpAPI,
`engine=google_images`).

---

# Current Verification Logic

- **Entry point**: `NemotronVisionModel.verify_product_match(image_path,
  product_name)` → `ProductMatchVerdict` (migrated from `QwenVisionModel`
  this session, via a failed Mistral attempt - see items 17-18 in
  Problems Already Solved).
- **Always receives the ORIGINAL, unmodified ERP product name** — never a
  normalized or retailer-specific version — regardless of which retailer or
  search path found the candidate.
- **Prompt** (`prompts.py: PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT`):
  redesigned once already this session (see Problems Already Solved) from a
  rigid field-by-field checklist to a holistic test: *"if a customer
  ordered the requested product and received the pictured retailer product
  instead, would they feel they got exactly what they wanted, or the wrong
  item?"* Explicitly instructs the model NOT to do literal text/substring
  comparison. Tolerates: word order, omitted/added marketing words,
  house-brand-vs-product-line naming differences (worked example baked in:
  "Gemini Dust Tea" ↔ "Tata Tea Gemini"), abbreviation expansion, unit
  formatting, capitalization/punctuation. Rejects: different product
  type/category, different named flavor/variant, different weight,
  different manufacturer — with worked examples baked in (Kohinoor Royal
  vs. Pulao Basmati Rice; Kellogg's Chocos Fills vs. Muesli; Chocolate vs.
  Plain Condensed Milk).
- **Acceptance gate** (`retailer_gallery_stage.py`): a retailer is accepted
  only if **both** `verdict.is_match is True` **and**
  `verdict.confidence >= 0.6` (`_DEFAULT_MIN_MATCH_CONFIDENCE`, a
  configurable constructor parameter). The confidence floor is an extra
  safety net on top of the model's own boolean judgment, since the JSON
  schema has no "none of the above"/partial-match option.
- **[HISTORICAL - Qwen-specific, model since migrated] Live-confirmed real
  behavior**: in an earlier session's end-to-end test, Qwen correctly
  found Amazon's top FunFoods Mayonnaise result to be the right
  brand/product-family but declined to confirm the 100g weight because it
  wasn't visibly legible in the photo (`is_match=False, confidence=0.95,
  reason="brand and product type match, but the weight is not
  visible..."`). This specific rejection is what motivated the later
  verifier-prompt rewrite (see Problems Already Solved) so an unreadable
  weight no longer causes a rejection - re-verified live afterward on
  Qwen (that same case flipped to `is_match=True`), and **re-verified
  again on the exact same real image against Nemotron 3 Nano Omni** (see
  item 18): `is_match=True, confidence=0.95`, correctly treating the
  unreadable/unspecified weight as non-contradicting.

---

# Current Storage Layout

**Simplified this session**: single root, `images/`, containing only final,
persisted per-product folders. There is no `images/downloads/` and no
`images/output/` anymore — both were removed as part of a deliberate
storage-architecture cleanup (see Important Design Decisions). Candidate
images that haven't been accepted/selected yet never touch `images/` at
all — both stages download them into an ephemeral OS temp directory
(`tempfile`) that's always cleaned up (win or lose) once the run/attempt
finishes.

```
images/
└── <slugify(product_name)>/
    ├── image_01.<ext>                       # RetailerGalleryStage result: full gallery, original order
    ├── image_02.<ext>
    ├── ...
    ├── front.<ext>                          # ProductImageSelectionStage (fallback) result instead
    ├── back.<ext>
    └── metadata.json                        # ALWAYS written, shape differs by which stage produced the result
```

A product folder contains EITHER the `image_NN.<ext>` gallery files (if
`RetailerGalleryStage` accepted a retailer) OR the `front.<ext>`/`back.<ext>`
files (if it fell back to `ProductImageSelectionStage`) — never both, since
only one of the two stages ever persists a given product.

`config.Settings.images_dir` is the only directory field left on `Settings`
(`downloads_dir`/`output_dir` were removed). `LocalFilesystemStorage`
resolves `product_dir(product_name)` directly under `images_dir`.

`metadata.json` content (`ProductImageResult`, serialized via
`model_dump_json`):

- **Always present**: `product_name`, `provider_used` (dict), `metadata`
  (dict, see below), `created_at`.
- **Retailer-gallery-mode result** (`front_image_path`/`back_image_path`/
  `front_confidence`/`back_confidence` all `null`):
  `metadata` contains: `retailer`, `retailer_product_url`,
  `search_query_used` (the actual string searched — original ERP name for
  Amazon/Flipkart, normalized for Zepto/Blinkit), `gallery_image_count`,
  `gallery_image_paths` (list of strings), `verification_confidence`,
  `verification_reason`, `brand`, `product_name` (as read by the vision
  model off the packaging — may differ from the top-level ERP
  `product_name`), `weight`.
- **Fallback-mode result** (`front_image_path`/`back_image_path`/
  `front_confidence`/`back_confidence` populated):
  `metadata` contains `front`/`back` sub-dicts each with `confidence`,
  `brand`, `product_name`, `weight`, `model_used`.

`utils/slugify.py:slugify()` is used everywhere a product name becomes a
folder name (lowercase, non-alphanumeric runs → single underscore).

---

# APIs Used

- **NVIDIA Build (NIM)** — `https://integrate.api.nvidia.com/v1/chat/completions`,
  OpenAI-compatible chat completions. Used for ALL vision-language calls
  (`classifier/nemotron.py`, migrated from Qwen this session via a failed
  Mistral attempt - see items 17-18 in Problems Already Solved). Auth:
  `Authorization: Bearer <NVIDIA_API_KEY>`.
- **SerpAPI** — `https://serpapi.com/search.json`. Two distinct engines used:
  - `engine=google_images` (`search/serpapi.py`, `SerpApiImageSearchProvider`)
    — used by the fallback stage (`selection_stage.py`).
  - `engine=google` (regular web search, `search/retailer_web_search.py`)
    with a `site:<domain>` filter — used for Zepto/Blinkit's search step
    only.
  Auth: `api_key` query parameter.
- **Retailer websites** (Amazon.in, Flipkart.com, Zepto.com, Blinkit.com) —
  no API, driven via Playwright (headless Chromium) directly: native search
  automation for Amazon/Flipkart, direct product-page navigation for all
  four (gallery extraction).

---

# Environment Variables

Read via `config.py`'s `Settings` (pydantic-settings, `env_file=".env"`,
`env_prefix="PIR_"` by default — **except** where a field overrides it):

| Variable (as set in `.env`) | Settings field | Notes |
|---|---|---|
| `NVIDIA_API_KEY` | `Settings.nvidia_api_key` | **No `PIR_` prefix** — uses `Field(validation_alias="NVIDIA_API_KEY")` to bypass the global prefix, matching NVIDIA's own convention and `classifier/nemotron.py`'s (originally `classifier/qwen.py`'s) design. This was a real bug fixed in an earlier session (see Problems Already Solved). |
| `PIR_SERPAPI_API_KEY` | `Settings.serpapi_api_key` | Normal `PIR_`-prefixed field. Confirmed working via the real `.env` file. |
| `PIR_GOOGLE_API_KEY`, `PIR_GOOGLE_CSE_ID`, `PIR_BING_API_KEY`, `PIR_BRAVE_API_KEY`, `PIR_OPENAI_API_KEY`, `PIR_ANTHROPIC_API_KEY`, `PIR_GOOGLE_GENAI_API_KEY` | Corresponding `Settings` fields | Declared but **unused** by any active code path — placeholders from the original generic-architecture design. |

Other `Settings` fields (directory paths, tuning knobs like
`max_queries`/`min_confidence`/`request_timeout_seconds`) exist from the
original architecture; only `images_dir` and `request_timeout_seconds` (via
`NemotronVisionModel`'s 60-second timeout floor calculation) are actually read
by the current active code paths. `downloads_dir`/`output_dir` were removed
from `Settings` entirely this session (see Current Storage Layout) —
candidate downloads now use ephemeral OS temp directories instead of a
config-driven scratch path.

A real `.env` file exists in the project root with working values for both
`NVIDIA_API_KEY` and `PIR_SERPAPI_API_KEY` (confirmed multiple times this
session, including one fully-live end-to-end API call to each service).

---

# Models Used

- **Vision-Language Model**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`,
  hosted on **NVIDIA Build**, defined as `classifier/nemotron.py:_MODEL_NAME`.
  **Migrated from** `qwen/qwen3.5-122b-a10b` this session (`qwen/qwen3.5-122b-a10b`
  hit end-of-life on NVIDIA Build - confirmed via a live `HTTP 410`
  response), via a first attempt at `mistralai/mistral-medium-3.5-128b`
  that never responded to any live request across two full sessions and
  was fully removed - see items 17-18 in Problems Already Solved for the
  complete writeup. **Fully live-verified**, unlike the abandoned Mistral
  attempt: text request, image request, production verifier prompt
  against real matching/non-matching products, and a full real retailer
  pipeline run (Amazon accepted and persisted; Flipkart and Zepto
  separately confirmed) all passed.
  - Request includes `"chat_template_kwargs": {"enable_thinking": false}` —
    the SAME mechanism Qwen used (unlike Mistral, which used a different
    `reasoning_effort` field) - confirmed live for this model's hosted
    NVIDIA Build endpoint (an `extra_body`-wrapped form, shown in this
    model's self-hosted-NIM docs, was tried first and rejected with a
    fast `HTTP 400`).
  - `temperature: 0.0`, `max_tokens: 500` (unchanged from Qwen).
  - Max images per single comparative request: **5** — verified live with
    a real 5-image request (NVIDIA's docs did not state an explicit
    number for this specific model, unlike Qwen's).
- **Search**: no LLM — SerpAPI (Google's actual search index) and each
  retailer's own native search engine (Amazon/Flipkart) are used as-is, no
  model involved in the search step itself.
- **Query normalization**: no LLM — deliberately rule-based (see Important
  Design Decisions).

---

# Problems Already Solved

1. **Google Images scraper bot-detection** (`search/google_images.py`,
   legacy/unused now): spent significant effort fixing CSS selectors
   (`div#islrg` → `div.bFtXbb`, etc.) before discovering the REAL root
   cause was Google's "unusual traffic" bot-check specific to this
   sandbox's network — confirmed via saved screenshot/HTML dump showing a
   literal reCAPTCHA challenge page. Not necessarily an issue on a normal
   residential network. This module is no longer in the active path anyway
   (superseded by SerpAPI for the fallback stage, and by retailer-first for
   the primary path).

2. **NVIDIA Qwen 60-second timeout / empty completions**: root-caused via
   web research (NVIDIA NIM docs, vLLM GitHub issues) to Qwen3.5's
   "thinking"/reasoning mode being enabled by default and consuming the
   entire token budget on hidden reasoning tokens, leaving visible
   `message.content` empty. Fixed by adding
   `"chat_template_kwargs": {"enable_thinking": false}` to every request.
   Also switched model from 397B to 122B variant due to separate
   forum-reported reliability issues with the larger model.

3. **`config.py` not exposing `NVIDIA_API_KEY`**: root cause was the global
   `env_prefix="PIR_"` not matching the bare `NVIDIA_API_KEY` name actually
   used in `.env` (and in `classifier/qwen.py`'s original os.environ-based
   fallback). Fixed with `Field(validation_alias="NVIDIA_API_KEY")` on that
   one field, bypassing the prefix specifically for it.
   `QwenVisionModel.__init__` simplified to read `self._settings.nvidia_api_key`
   directly, removing the old `getattr(..., "nvidia_api_key", None) or
   os.environ.get(...)` fallback chain.

4. **Overly strict product-match verification prompt**: user found
   "GEMINI DUST TEA 500g" (ERP) vs. Amazon's "Tata Tea Gemini 500g" was
   wrongly REJECTED. Root cause: the original prompt's rigid,
   field-by-field-in-priority-order checklist framing primed the model
   toward literal comparison. Redesigned to a holistic "would the customer
   feel wronged" test, with the Gemini/Tata example and several others
   baked directly into the prompt as few-shot anchors.

5. **Flipkart gallery-scraping contamination**: naive DOM scanning (all
   `<img>` tags, or climbing to a shared ancestor `<div>`) picked up
   unrelated "similar products" carousel images sharing the exact same CSS
   classes/size bucket as the real gallery. Solved by using Flipkart's own
   JSON-LD structured data instead — though even THAT was found to
   sometimes contain a genuinely mismatched image due to Flipkart's own
   real catalog data-quality issues (documented as a caveat, not "fixed"
   since it's not our bug — verification is the actual safety net for it).

6. **Amazon/Flipkart native search automation bugs** (this session):
   - Amazon: pressing Enter after typing did not reliably submit the search
     — fixed by clicking the actual submit button instead.
   - Flipkart: a "Login" popup appears on page load and intercepts clicks
     on the search box — fixed by pressing `Escape` before interacting
     (verified live, does not depend on hashed CSS classes).
   - Flipkart: the search-box placeholder selector matched TWO inputs (one
     `readonly` duplicate) — fixed with `:not([readonly])`.
   - Both: raw `href` attributes on result links are relative paths, not
     absolute URLs — fixed by resolving against each site's homepage URL.

7. **Retailer priority-list churn**: went through several iterations
   (Amazon/Flipkart/BigBasket/Zepto in the original POC phase → user asked
   to replace one entry with Zepto → later asked to replace BigBasket
   entirely and add Blinkit, landing on the current Amazon → Flipkart →
   Zepto → Blinkit order, explicitly confirmed with the user).

8. **[FULLY REVERTED BY ITEM 10 - kept for history/diagnosis value only]
   Retailer search was losing correct products for many FMCG items**
   (this session) — the query-broadening/candidate-pooling/ranking design
   below was entirely rolled back (see item 10) after it caused a real
   accuracy regression on other products; the ROOT-CAUSE DIAGNOSIS in this
   item is still accurate and useful context, but none of the code changes
   it describes are in the current implementation anymore.
   User reported broad failures across products like
   "JERSEY KESAR FLAVOURED MILK 185ml PET" and "KISSAN FRESH TOMATO
   KETCHUP 950g PCH", where the retailer genuinely had the right product
   but under a differently-worded title. **Root cause, diagnosed BEFORE
   any code was changed** (confirmed live, not guessed): every retailer
   path collapsed to exactly ONE candidate before verification ever saw
   it — `amazon.py`/`flipkart.py`'s native search returned only the first
   organic result's URL (no title even extracted), and
   `retailer_web_search.py`'s SerpAPI path discarded every result but the
   first same-domain match, even though SerpAPI's response already
   included a `title` per result. Live evidence gathered this session:
   Amazon's #1 organic result for "JERSEY KESAR FLAVOURED MILK 185ml PET"
   was **"Mother Dairy Flavoured Milk, Kesar Elaichi, 180 ml"** - a
   different brand entirely - while the real Jersey-brand product sat at
   position #5; Flipkart's naive "first `a[href*="/p/"]`" for "KISSAN
   FRESH TOMATO KETCHUP 950g PCH" grabbed an unrelated "Little Joys"
   product from an empty-text image-wrapper anchor sharing its href with
   the real (differently-positioned) title anchor. **Verification itself
   was never the problem** — every prior live test showed it correctly
   telling right from wrong once given a candidate; it just was almost
   never given more than one, frequently-wrong, candidate per retailer.
   **Fix** (verification/threshold logic completely untouched):
   - `search/query_normalizer.py`: added `generate_broadened_queries()`
     (full normalized query → same minus quantity/unit → first 4 words of
     that) and extended abbreviation expansions with `PET`→"Bottle",
     `PCH`→"Pouch" (moved out of the old noise-word delete-list - a real
     listing plausibly contains the word "Pouch"), `BTL`→"Bottle",
     `CAN`→"Tin". ALL four retailers now go through this (previously only
     Zepto/Blinkit's SerpAPI path was normalized; Amazon/Flipkart's native
     search got the raw ERP string on an assumption that turned out not to
     hold broadly).
   - `search/similarity.py` (new): dependency-free, word-order- and
     unit-spacing-invariant token-overlap similarity
     (`title_similarity`/`rank_candidates`) - deliberately simple/rule-based
     like the normalizer, used ONLY to order candidates, never to decide
     correctness.
   - `search/retailers/amazon.py` / `flipkart.py`:
     `search_product_page()` → `search_product_candidates()`, returning up
     to 5 `SearchCandidate(url, title)` pairs instead of one bare URL - new,
     live-verified title selectors (see their docstrings / the provider
     bullets above).
   - `search/retailer_web_search.py`: `find_product_page()` /
     `find_retailer_product_page()` → `find_product_page_candidates()` /
     `find_retailer_product_candidates()`, returning up to 5 candidates
     using SerpAPI's already-present `title` field (no new scraping
     needed for Zepto/Blinkit).
   - `retailer_gallery_stage.py`: `_try_retailer` rewritten to gather up
     to 5 unique candidates per retailer (trying broader queries only
     until the pool is full), rank them against the ORIGINAL product name,
     and try each in ranked order - stopping at the first accepted one -
     before moving to the next retailer. `VERIFICATION_CONFIDENCE_THRESHOLD`,
     `verify_product_match`, and the accept/reject condition are all
     byte-for-byte unchanged.
   - Added detailed logging throughout: normalized query, generated
     broadened queries, each query's returned candidate titles, the full
     ranked list with similarity scores, and per-candidate
     accept/reject/error results with reasons.
   **Verified live end-to-end** (real API calls, not mocked): re-ran
   "KISSAN FRESH TOMATO KETCHUP 950g PCH" through the real pipeline after
   the fix. Query 1 ("...950g Pouch") returned only 1 Amazon candidate;
   the pool correctly broadened to query 2 ("...Pouch"), which returned 4
   more, filling the pool to 5. Candidate #1 (highest similarity, "Doy
   Pack, 950g...") was correctly REJECTED by the unchanged verifier for a
   genuine weight mismatch (900g visible vs. 950g requested); candidate #2
   ("Kissan Fresh Tomato Ketchup Pouch, 850gm Pouch") was tried next and
   ACCEPTED (confidence 0.95) - the loop stopped immediately, candidates
   #3-5 were never touched. 4 images persisted correctly with the new
   `matched_candidate_title` / `candidate_similarity_score` metadata
   fields populated. Also confirmed offline against the real titles
   scraped live from both Amazon and Flipkart for both example products:
   the correct-brand/correct-product candidate ranks #1 in both cases,
   ahead of the old code's blindly-trusted (and wrong) first result.

9. **[SUPERSEDED BY ITEM 10 - kept for history only] Retailer
   simplification, Brand column, candidate pre-filtering, and NVIDIA
   retry** (this session, second iteration on the search redesign) - of
   everything below, only the retailer removal/order and the NVIDIA retry
   logic are still in effect; Brand/filtering/ranking were fully reverted
   in item 10 after a real accuracy regression - see item 10 before
   assuming anything below reflects the current implementation.
   User tested item 8's fix on a variety of real products and asked for
   four further changes, all implemented while keeping verification
   completely untouched:
   - **Removed Zepto, Blinkit, and BigBasket entirely** (provider modules
     `search/retailers/zepto.py` / `blinkit.py` deleted; `search/retailer_web_search.py`
     deleted too - it existed ONLY to drive their SerpAPI-based search, so
     once they were gone it was fully dead code; `ZEPTO`/`BIGBASKET`
     removed from `config.py`'s `SearchProviderKey` enum; every remaining
     Zepto/Blinkit docstring mention in `interfaces.py`, `search/__init__.py`,
     `search/retailers/__init__.py` updated). Retailer priority is now
     simply **Amazon → Flipkart → Google Images (fallback)**. Both
     remaining retailers use native search only - `RetailerGalleryStage`
     no longer needs (or accepts) a `serpapi_api_key` constructor
     parameter at all.
   - **Brand column support**: `batch/column_detection.py` gained an
     OPTIONAL Brand alias set (`Brand`, `Brand Name`, `Product Brand`,
     `Manufacturer(/Brand)`) - optional because, unlike EAN/Description,
     its absence is not an error. `batch/models.py:ProductRow` gained a
     `brand: str = ""` field; `batch/excel_import.py` extracts it when
     present. `RetailerGalleryStage.run()` gained a new, additive
     `brand: str | None = None` parameter (defaults preserve the exact old
     behavior for the single-product CLI/UI, which never passes it);
     internally it builds `effective_name = f"{brand} {product_name}".strip()`
     and uses THAT for query generation, candidate ranking, and vision
     verification - while `product_name` (plain, unbranded) is still what
     gets used for the output folder name and the Google Images fallback,
     unchanged. `batch/processor.py` now calls
     `stage.run(row.product_description, brand=row.brand or None)`.
   - **Cheap pre-verification candidate filtering** (`search/candidate_filter.py`,
     new): extracts brand-presence, a curated flavour/variant keyword, and
     weight/volume from a candidate's TITLE TEXT ONLY, and rejects a
     candidate ONLY on a confident mismatch - directly addresses the
     user's complaint that obviously-wrong-brand candidates (Amul, Mother
     Dairy, Britannia, Prayag, for a Jersey-brand query) were reaching the
     vision model at all. Deliberately ASYMMETRIC/conservative: ambiguous
     cases (no extractable weight, no recognized flavour keyword, no brand
     supplied) always pass through untouched - a false rejection here
     permanently removes a candidate before verification ever sees it,
     which would be worse than the original problem. The weight-mismatch
     threshold is a >2x ratio, NOT exact equality - verified live that
     Qwen itself accepted a real "850gm" candidate against a "950g"
     request (ratio ~1.12) as a genuine match, so an exact/tight
     comparison would have wrongly pre-rejected a candidate the vision
     model was actually happy with. Wired into `retailer_gallery_stage.py`
     between candidate-gathering and ranking (`_filter_candidates`, called
     from `_try_retailer` before `rank_candidates`).
   - **NVIDIA Build retry logic** (`classifier/qwen.py:_classify`): now
     retries transient failures (HTTP 500s, malformed/missing/empty
     `message.content`, timeouts, network errors) up to 2 times (3 attempts
     total) with a 2-second backoff before raising - these are API
     failures, not a verdict, so `retailer_gallery_stage.py`'s existing
     "move to the next candidate" behavior on an exception now happens
     only AFTER retries are exhausted, and the log message explicitly says
     "vision API failure (not a product mismatch)" to distinguish it from
     an actual `is_match=False` rejection. `VisionModelAuthError` (bad API
     key) is deliberately NEVER retried - retrying can't fix an invalid
     credential. All three of `classify_image`/`select_best_candidate`/
     `verify_product_match` benefit, since they share this same low-level
     `_classify` method.
   **Verified live end-to-end** (real API calls, not mocked), twice:
   (1) "Jersey" + "KESAR FLAVOURED MILK 185ml PET" - Amazon returned 5
   candidates, ALL 5 wrong-brand (Amul/Mother Dairy), ALL eliminated
   pre-verification with ZERO vision calls; Flipkart returned 5, 4
   wrong-brand plus one correct-brand-wrong-flavour ("Godrej Jersey Recharge
   **Mango**" vs. requested **Kesar**) - all 5 eliminated, again ZERO
   vision calls; correctly fell back to Google Images, which itself hit
   two real transient NVIDIA failures in a row on the front-image
   comparative call, retried both, and (after exhausting retries on that
   specific call) correctly logged it as a failure and moved on rather
   than treating it as a rejection, still successfully persisting the back
   image. (2) "Kissan" + "FRESH TOMATO KETCHUP 950g PCH" - Amazon returned
   5 candidates, one wrong-brand ("VEEBA") correctly eliminated
   pre-verification, the other 4 ranked and the top one tried; that single
   vision call hit a REAL live HTTP 500 ("invalid type: unit variant..."),
   was automatically retried, and succeeded on attempt 2 with
   `is_match=True confidence=0.95` - accepted and persisted correctly. Also
   confirmed offline: brand-mismatch rejection, flavour-mismatch rejection,
   the 850g-vs-950g pass-through (matching the real verified-accepted
   case), an order-of-magnitude weight rejection, and that missing/ambiguous
   signals never cause a rejection.

10. **FULL REVERT of items 8 and 9's search/candidate-selection changes**
    (this session, immediately after item 9) - after testing item 9's
    Brand-aware/filtering/ranking behavior on "a variety of real products,"
    the user found it caused **real accuracy regressions**: some products
    that previously matched correctly started returning WRONG products.
    Rather than debug which specific mechanism (brand combination
    confusing verification, filtering incorrectly admitting a wrong
    candidate while discarding a correct one, ranking prioritizing a
    plausible-but-wrong title, etc.) was responsible, the user asked for a
    complete, unconditional revert of ALL retrieval/search logic changes
    from items 8 and 9, keeping ONLY: Zepto/Blinkit/BigBasket removed, the
    Amazon → Flipkart → Google Images order, and the NVIDIA retry logic.
    **Everything else was restored to the EXACT pre-item-8 implementation**
    (reconstructed from this conversation's own transcript, not
    reconstructed from guesswork):
    - `search/retailers/amazon.py` / `flipkart.py`: `search_product_candidates()`
      reverted back to the original `search_product_page(page, query) -> str | None`
      (single URL, no title extraction, original selectors -
      `a[href*="/dp/"]` / bare `a[href*="/p/"]` - including the original
      Flipkart wrapper-anchor limitation this project had previously found
      and fixed; see the Flipkart provider bullet above for that caveat).
    - `search/query_normalizer.py`: `generate_broadened_queries()` /
      `_strip_quantity()` removed entirely; `_ABBREVIATION_EXPANSIONS`
      reverted to the original set (no `PET`/`BTL`/`CAN`); `PCH` moved back
      into `_NOISE_WORDS` (removed from search text, not expanded to
      "Pouch"). `normalize_for_search()` itself is unchanged and still
      computed/logged in `retailer_gallery_stage.py`, but - since both
      remaining retailers are native/raw-name searches - it isn't actually
      applied to any search call, exactly matching the pre-item-8 reality
      once Zepto/Blinkit (its only real consumers) were gone.
    - `search/candidate_filter.py` and `search/similarity.py` - **deleted
      entirely** (both were fully superseded; `search/retailers/__init__.py`'s
      `SearchCandidate` dataclass removed too, since nothing constructs one
      anymore).
    - `retailer_gallery_stage.py`: `_try_retailer` reverted to a single
      linear attempt per retailer (search -> extract -> download first
      image -> verify -> accept/reject), no candidate pooling, no
      filtering, no ranking, no `brand` parameter on `run()`. `_persist`
      reverted to its original metadata shape (`search_query_used` field
      restored; `matched_candidate_title` / `candidate_similarity_score`
      removed).
    - `batch/column_detection.py` / `models.py` / `excel_import.py`: Brand
      column detection/field/extraction removed entirely -
      `detect_columns()` is back to its original `(ean_index,
      description_index)` 2-tuple return, `ProductRow` has no `brand`
      field.
    - `batch/processor.py`: back to calling `stage.run(row.product_description)`
      with no `brand` kwarg.
    - **Kept, per explicit instruction**: the `_RETAILERS` list (Amazon +
      Flipkart only, no `domain`/`path_markers`/`serpapi_api_key` - those
      were only ever needed for Zepto/Blinkit's SerpAPI path), and
      `classifier/qwen.py`'s NVIDIA Build retry logic
      (`_MAX_VISION_RETRIES` / `_VISION_RETRY_BACKOFF_SECONDS` in
      `_classify()`) - completely untouched by this revert.
    **Verified live end-to-end** (real API calls): re-ran "KISSAN FRESH
    TOMATO KETCHUP 950g PCH" through the reverted pipeline - logs show
    exactly the original single-candidate flow (one product page found on
    Amazon, no candidate-list/filtering/ranking log lines at all), hit two
    more real transient NVIDIA failures (confirming the retry logic is
    still fully intact), succeeded on attempt 3, and persisted 6 images
    with the original `search_query_used`-shaped metadata (no
    `matched_candidate_title` / `candidate_similarity_score` fields).
    Confirmed offline that `batch/excel_import.py` no longer exposes a
    `brand` field on `ProductRow`. **Do not re-introduce any of items 8/9's
    reverted behavior (brand-aware queries, candidate pooling/ranking,
    title-based pre-filtering) without the user explicitly asking again -
    it was tried, tested against real products, and explicitly rolled back
    for a real accuracy regression, not a style preference.**

11. **Always include out-of-stock results before scraping** (this session,
    a narrow, isolated addition - no candidate selection, normalization,
    verification, threshold, or provider-interface changes). Both
    `search_product_page(page, query) -> str | None` signatures are
    unchanged; only their internal implementation gained one extra step
    between "search results loaded" and "start reading result cards."
    - **Amazon**: URL parameter, not a Playwright interaction. Verified
      live: the sidebar's "Include Out of Stock" link has
      `href=".../s?k=<query>&rh=p_n_availability%3A1318485031&..."`, and
      that `rh=p_n_availability:1318485031` value was confirmed IDENTICAL
      across two unrelated queries ("tomato ketchup" and "jersey kesar
      flavoured milk") - a stable, query-independent filter ID, not
      per-search markup. Chosen over a Playwright click because it's
      simpler and more reliable (no extra element-find/click/wait cycle,
      no risk of a changed sidebar layout breaking a selector) - exactly
      what implementation note 1 asked to prefer when available. New
      `_add_out_of_stock_filter()` helper appends this via
      `urllib.parse` (no manual string concatenation); `search_product_page()`
      now does `page.goto()` to the filtered URL and re-waits for result
      cards AFTER the normal search loads, BEFORE any result card is read.
      Verified live: the returned candidate URL itself carried
      `refinements=p_n_availability%3A1318485031`, confirming Amazon
      registered the filter as applied to that search.
    - **Flipkart**: Playwright interaction attempted, not a URL parameter -
      because there's currently no such parameter (or filter at all) to
      use. Verified live across two different categories (FMCG groceries
      "tomato ketchup" and electronics "laptop"): Flipkart's only
      "Availability" filter option in both cases was "GST Invoice
      Available" - no out-of-stock/unavailable toggle exists in the
      current UI. Per requirement 4 ("if a retailer does not expose such a
      filter... continue with the normal results instead of failing"), new
      `_apply_out_of_stock_filter_if_present()` does a real, scoped
      text-based search (`out of stock|unavailable`, case-insensitive,
      excluding matches inside a product link so an individual listing's
      own "Out of Stock" badge is never mistaken for a filter control) and
      clicks the first real match if one exists; today it always finds
      none and returns immediately, so normal scraping proceeds
      unaffected - but this is a genuine, forward-compatible search rather
      than a hardcoded skip, so it starts working automatically if
      Flipkart ever adds such a filter.
    **Verified live end-to-end** (real API calls): called both
    `search_product_page()` functions directly for "KISSAN FRESH TOMATO
    KETCHUP 950g PCH" - Amazon's final page URL and the returned product
    URL both carried the out-of-stock filter parameter, and it still
    correctly found the same "Kissan-Fresh-Tomato-Ketchup-950g" product as
    before; Flipkart's filter search completed harmlessly (nothing found)
    and it still correctly found the same "kissan-fresh-tomato-sauces-ketchup"
    product as before. A subsequent full-pipeline run showed Amazon/Flipkart
    both being tried and failing this time - confirmed via logs to be
    ordinary, pre-existing flakiness (a genuine Qwen weight-mismatch
    rejection on Amazon, then two real transient NVIDIA 500s exhausting
    retries on Flipkart, then a real SerpAPI network timeout in the
    fallback) entirely unrelated to this change - the search/filter step
    itself worked correctly in both cases before verification ran.

12. **[SUPERSEDED BY ITEM 13 - kept for history only] Zepto re-added as a
    first-class retailer, first attempt (SerpAPI-backed search)**. The
    user explicitly rejected this approach in the very next message ("Do
    not use SerpAPI, Google Search, Google site: queries, or any other
    external search engine... I want Zepto to behave exactly like Amazon
    and Flipkart by using Zepto's own search functionality") and asked for
    a from-scratch live re-investigation of whether Zepto's OWN search is
    actually usable. It is - see item 13. `zepto.py`'s `search_product_page`
    no longer calls SerpAPI at all; none of the SerpAPI-specific claims
    below reflect the current code.
    Original (now-superseded) writeup, between
    Flipkart and the Google Images fallback, exposing the exact same
    `search_product_page(page, query) -> str | None` /
    `extract_gallery(page, product_url) -> GalleryResult` interface as
    Amazon/Flipkart. Amazon, Flipkart, normalization, Qwen verification,
    confidence thresholds, and existing retailers' candidate-selection
    logic were not touched - the only edits to shared files were adding one
    `_RETAILERS` entry + one import in `retailer_gallery_stage.py`, plus
    docstring updates. New module: `search/retailers/zepto.py`.
    - **Investigated live first, per the "never guess selectors/endpoints"
      rule**: Zepto's homepage and its own `/search?query=...` page both
      confirmed stuck (React/Next.js app never hydrates in this sandbox -
      `document.title` stays generic, `window.location.href` doesn't
      change, and `read_network_requests` showed zero XHR/fetch calls
      fired regardless of URL). Driving Zepto's own search box the way
      Amazon/Flipkart's is driven is therefore not possible here.
    - **Search**: uses the SerpAPI transport already trusted elsewhere in
      this project (`search/serpapi.py`'s endpoint, but the regular
      `engine=google` instead of `engine=google_images`) with a
      `site:zepto.com/pn/` scope - a documented, authenticated API call,
      not a scraper or a guessed endpoint. Verified live against two real
      ERP names: "KISSAN FRESH TOMATO KETCHUP 950g PCH" and "JERSEY KESAR
      FLAVOURED MILK 185ml PET" both correctly resolved to their real
      Zepto product pages. Getting there required query cleanup beyond
      what raw ERP text supports for a literal Google index (unlike
      Amazon/Flipkart's own fuzzy search): the raw string returned ZERO
      SerpAPI results in both cases (confirmed via SerpAPI's own "Google
      hasn't returned any results" error). Fixed by (1) reusing the
      existing, **unmodified** `search.query_normalizer.normalize_for_search()`
      (it's explicitly designed "for the string sent to a retailer's own
      search box" - this is reuse, not a normalization change), then (2) two
      new, Zepto-search-only, self-contained helpers in `zepto.py` itself:
      `_strip_trailing_quantity` (drops a trailing "950g"-style token -
      verified live that the attached form returns 0 results, a spaced
      form returns 1, and dropping it returns 39) and
      `_strip_packaging_noise` (drops PET/BTL/CAN - the same packaging-code
      category as `query_normalizer`'s own noise-word list, but not
      extended there since that module is shared/off-limits; PET
      specifically re-verified live as the blocker for the milk product).
    - **Product pages / gallery extraction**: same DOM-scraping approach as
      this project's original (later-deleted) Zepto implementation -
      re-verified live this session: direct navigation to a known product
      URL server-renders a product-specific `document.title` even though
      the visual body stays blank, and the DOM already contains every
      gallery `<img>` tag (32 tags on one product, 16 matching the
      `/cms/product_variant/` marker = 8 unique images). Images are served
      through an ImageKit-style CDN; stripping the `/tr:<params>/` resize
      segment was re-verified live to yield a genuinely larger original
      file (3,338 bytes → 57,103 bytes, both HTTP 200).
    - **New finding, not in the historical implementation**: a bare
      Playwright page with no explicit user agent - exactly what
      `retailer_gallery_stage.py` hands every retailer - gets HTTP 429 from
      Zepto on product-page navigation, reproduced live on two different,
      previously-unvisited URLs and persisting after a 30s wait (not a
      brief burst limit). A realistic desktop Chrome user agent + viewport
      on a *separate* browser context got a clean HTTP 200. Since
      `retailer_gallery_stage.py` shares one `page`/context sequentially
      across all three retailers, `extract_gallery` opens its own
      short-lived `BrowserContext` (via `page.context.browser.new_context(...)`)
      instead of navigating on the shared `page` - Amazon and Flipkart run
      first and always finish their own turns before this context is ever
      created, so they're never touched by it.
    - **Verified live end-to-end** (real SerpAPI + Playwright calls, no
      mocking): both test products correctly found their real Zepto
      product page and extracted a real, deduplicated original-resolution
      image gallery (8 images and 5 images respectively).
    - **Limitations, reported honestly**: search quality depends on
      Google's index of Zepto's catalog, not Zepto's own search ranking -
      a real, unindexed product returns `None` (pipeline correctly falls
      through to Google Images, as required). No location/serviceability
      check is possible (Zepto requires a delivery location for live
      stock/pricing, and the app never hydrates here) - a returned page may
      be out of stock in a given area; this is the same trade-off already
      accepted for Amazon/Flipkart, where downstream Qwen verification -
      not availability - is what actually gates acceptance. No login/cookie
      handling is needed or performed.

13. **Zepto's own native search, replacing item 12's SerpAPI approach**
    (this session). Same interface, same `_RETAILERS` wiring, same
    "don't touch Amazon/Flipkart/normalization/Qwen/thresholds"
    constraints as item 12 - only `zepto.py`'s internals changed.
    `search/serpapi.py` (used elsewhere for the Google Images fallback,
    unchanged) is no longer referenced by `zepto.py` at all.
    - **Re-investigated live from scratch**, per explicit instruction, why
      Zepto's own search previously looked unusable. Root cause: item 12's
      investigation used a BARE Playwright context (no user agent) - the
      exact thing that also gets a 429 from Zepto's AWS WAF (see item 12's
      gallery-extraction finding). With a realistic desktop Chrome UA +
      viewport on its own context, the homepage hydrates completely: real
      body text, ₹ pricing, and genuine session cookies including an
      `aws-waf-token` - confirming this was bot mitigation all along, not
      a fundamental "can't run under Playwright" limitation.
    - **Delivery location**: Zepto auto-resolves one server-side per
      request (not from Playwright's `geolocation` context option, which
      was set but ignored - the API call's resolved coordinates didn't
      match it). That auto-resolved location was confirmed live to be
      serviceable. Manually setting a location via Zepto's own "Select
      Location" → address-autocomplete flow was tested live and is a real
      trap: picking a genuine, real, Google-indexed Delhi address put the
      whole session into an unserviceable "Sit Tight! We're Coming Soon!"
      state with zero catalog. No location is set for this reason - the
      server's own default is the only one confirmed safe.
    - **Search flow, fully native, verified live end-to-end**: load
      homepage → click the search-trigger text ("Search for over 5000
      products" is the input's real placeholder, only revealed after this
      click - the input is `type="hidden"` before it) → type the query →
      press Enter → read `a[href*="/pn/"]` links from the results. Backed
      by Zepto's own `POST bff-gateway.zepto.com/user-search-service/api/v3/search`,
      observed live carrying the typed query.
    - **Two real bugs found and fixed via live iteration, not guessed
      fixes**:
      1. *Input garbling*: typing immediately after the search-trigger
         click races with the site's own route-transition re-render and
         silently drops/reorders characters (e.g. "KISSAN FRESH TOMATO
         KETCHUP" became "ATO KETCHUP" on ~1 in 3 attempts). A fixed
         post-click delay (tested at 300ms/800ms) did not reliably fix
         it; waiting for the URL to settle on `/search` did not either.
         Fixed with `_type_and_verify`: type, read the input's value back,
         and retry (full clear + retype) until it matches - self-verifying
         rather than a timing guess, confirmed live to succeed within 2
         attempts across 6 consecutive real runs.
      2. *Decoy results*: before a query is even submitted, the search
         overlay already shows `/pn/` links for an unrelated "trending
         products" carousel. Waiting for "any product link" (the
         Amazon/Flipkart pattern) matches these decoys instantly - the
         real, submitted-query results replace them ~0.5-1s later. Fixed
         by snapshotting the decoy links before pressing Enter and
         waiting for the link set to actually change. This alone was
         still insufficient for one case: "MAGGI 2 MINUTE NOODLES MASALA"
         - the search API's own `totalProductCount` field reported 17,
         but the page rendered Zepto's own literal "Could not find any
         products for ..." message and the decoy set never changed. That
         field is NOT trusted anywhere in this module because of this;
         `_wait_for_real_results` polls for either a new (non-decoy) link
         OR that literal message, whichever comes first.
    - **Query cleanup, verified live to be necessary for Zepto's OWN
      search** (not just Google's, as item 12 had found): raw ERP text
      with a trailing quantity ("KISSAN FRESH TOMATO KETCHUP 950g")
      returned only 2 results and both were the WRONG product ("The True
      Tomato Ketchup"); dropping the quantity ("KISSAN FRESH TOMATO
      KETCHUP") returned 14 results with the correct product dominating
      the top. Reuses the existing, **unmodified**
      `search.query_normalizer.normalize_for_search()` (already designed
      "for the string sent to a retailer's own search box") plus one
      Zepto-local helper, `_strip_trailing_quantity`, for the trailing
      quantity specifically.
    - **Gallery extraction**: unchanged in mechanism from item 12 (DOM
      `<img>` scan for the `/cms/product_variant/` marker + ImageKit
      resize-prefix stripping) - still uses its own isolated, realistic
      `BrowserContext` per call for the same AWS WAF reason.
    - **Verified live end-to-end, repeatedly, real Playwright + real
      Zepto, no mocking**: "KISSAN FRESH TOMATO KETCHUP 950g PCH" and
      "FUNFOODS MAYO COMBO 100g" both correctly resolved to real,
      matching Zepto products with real image galleries, reproduced
      across multiple separate runs. "JERSEY KESAR FLAVOURED MILK 185ml
      PET" and "MAGGI 2 MINUTE NOODLES MASALA 70g PKT" both correctly
      resolved to `None` (Zepto's own search genuinely has no result for
      either, confirmed via the literal on-page message) - the pipeline
      falls through to Google Images for these, exactly as designed.
    - **Limitations, reported honestly**: Zepto's own search can return
      zero results for a real product even with a clean query (the two
      cases above) - this is a genuine limitation of Zepto's search/
      catalog, not a scraping defect, and is handled by falling through to
      the next retailer like any other retailer's miss. No location is
      pinned (see above), so results reflect whatever location the
      request auto-resolves to; downstream Qwen verification, not
      availability, is what actually gates acceptance - same trade-off
      already accepted for Amazon/Flipkart. No login/cookie handling is
      needed or performed - cookies are whatever the site sets itself on
      a normal, unauthenticated visit.

14. **Root-cause investigation: integrated pipeline logged "[Zepto] no
    product page found in search results" for a product confirmed to
    exist** (this session). User explicitly asked for proof via logs, not
    assumptions, and to identify the precise failure point before changing
    anything. Added instrumentation (kept in the code, not removed after):
    a "[%s] search_fn bound to %s.%s (module file: %s)..." log in
    `retailer_gallery_stage.py:_try_retailer` confirming exactly which
    file/function is invoked, plus detailed `logger.info`/`debug`/`warning`
    calls throughout `zepto.py` (raw vs. cleaned query, homepage load
    status, trigger/input click outcomes, typed-value verification,
    decoy-link snapshot, and the exact reason `_wait_for_real_results`
    returned empty).
    - **Confirmed NOT the cause**: a stale/wrong implementation. The log
      showed `search_fn bound to search.retailers.zepto.search_product_page
      (module file: .../search/retailers/zepto.py)` - the correct, current,
      native-search module - invoked with the correct, unmodified
      `product_name`.
    - **Confirmed NOT the cause**: query cleaning, selectors, or the
      decoy/no-results detection logic. The logged trace showed the query
      cleaned correctly (`'FUNFOODS MAYO COMBO 100g' -> 'FUNFOODS
      Mayonnaise'`), the homepage loading (`status=200`), the trigger and
      input clicking successfully, and the query typing verified correct
      on the first attempt.
    - **Actual root cause, found in the logged `body sample` at the exact
      point of failure**: after pressing Enter, the page rendered
      `"...Please Login\n\nOops! Please login to continue searching\n\n
      Login..."` instead of results - Zepto occasionally gates search
      results behind a login wall. This is a real, distinct page state
      that `_wait_for_real_results` did not previously recognize, so it
      fell through to the generic 20-second timeout and was logged
      identically to a genuine empty-catalog result.
    - **Proven transient/probabilistic, not deterministic** - per the
      explicit "do not assume, prove it" instruction: re-ran the IDENTICAL
      query standalone immediately after the failure - succeeded (16
      correct results). Then replayed the EXACT pipeline sequence (real
      Amazon search -> real Flipkart search -> Zepto search, all in one
      browser process, matching `RetailerGalleryStage.run()`'s shared-page
      pattern exactly) - also succeeded immediately. This rules out the
      retailer ordering, the shared browser/page, and the specific query
      as causes; the login wall is an occasional site-side anti-bot gate
      (consistent with the AWS WAF behavior already documented for this
      site in item 13), not a bug in this project's code.
    - **Fix applied, strictly diagnostic/detection-only - no search
      algorithm redesign, per explicit instruction**: added
      `_LOGIN_WALL_TEXT = "Please login to continue searching"` as a THIRD
      signal in `_wait_for_real_results`, checked before the existing
      "Could not find any products" no-results check. When seen, it logs a
      `WARNING` explicitly distinguishing this from a genuine empty result
      and returns immediately instead of waiting out the full timeout.
      Behavior (the function still returns `[]`/`None` on this path) is
      unchanged - only the speed and truthfulness of the logging changed.
      No retry-on-login-wall logic was added despite the transience
      evidence above; that would be a robustness/behavior improvement,
      which was explicitly out of scope for this task ("the objective is
      not to improve search quality") - **recommended as a follow-up if
      the user wants it**, e.g. retrying once with a fresh isolated
      context specifically when this wall is seen.
    - **The unrelated NVIDIA "HTTP 410" errors seen in the same trace**
      (during Amazon's and Flipkart's own verification attempts) are a
      separate, pre-existing kind of transient failure already handled by
      `classifier/qwen.py`'s existing retry logic - not investigated
      further here since they're outside this task's scope and did not
      prevent Zepto's turn from being reached.

15. **Zepto broad-search investigation for "JERSEY KESAR FLAVOURED MILK
    185ml PET" (this session)** - the user asked whether the target SKU
    (confirmed by them to exist and, separately, confirmed by this project
    to be in-stock/purchasable at its direct product URL) might be
    reachable via broader queries + local filtering, rather than
    concluding Zepto's search can't find it at all. Tested 5 broader
    queries live (`JERSEY MILK`, `GODREJ JERSEY`, `FLAVOURED MILK`, `KESAR
    MILK`, `JERSEY`), capturing every `/pn/` product card on each results
    page (scrolling to trigger lazy-loaded cards) - 135 total card
    observations across 24-40 unique products per query. The target SKU
    (`godrej-jersey-kesar-flavoured-milk`) appeared in **zero** of them -
    notably absent even from `GODREJ JERSEY` (which returned 6 *other*
    real Godrej Jersey dairy products: curd, ghee, paneer, toned/full-
    cream milk, doodhpeda) and from `FLAVOURED MILK`/`KESAR MILK` (which
    both returned *other* brands' Kesar-flavoured milks). Per the user's
    own stated decision criterion, this confirms Zepto's own search index
    does not expose this SKU for any reasonable native query, broad or
    narrow - not merely "the automation only takes the first result."
    A broaden-then-locally-filter retrieval strategy was explicitly NOT
    implemented, since its precondition (the product appearing somewhere
    in broader results) did not hold - implementing it anyway would have
    been exactly the kind of speculative workaround the user asked not to
    build. No code change to `zepto.py` from this investigation.

16. **Amazon regression investigation (this session)** - user observed
    `[Amazon] search_fn returned: None` / `[Amazon] no product page found
    in search results` and asked whether Amazon had changed. Live
    investigation with a bare Playwright context reproduced something
    surprising-looking at first: the homepage's first response is **HTTP
    202** (not 200) with a **completely empty body** and zero `<input>`
    elements. Response headers explain it precisely:
    `x-amzn-waf-action: challenge` - this is Amazon's own AWS WAF (the
    same bot-mitigation vendor already documented for Zepto) issuing a
    challenge. Confirmed live: the browser's own JS automatically fetches
    `challenge.js` / `inputs?client=browser` / `mp_verify` from
    `*.token.awswaf.com`, then silently re-requests the homepage, which
    returns a full HTTP 200 with real content ~5 seconds later - no
    CAPTCHA image, no login page, no DOM/selector change; `#twotabsearch
    textbox` is unchanged and still correct. This is NOT a regression -
    `search_product_page`'s `.click(timeout=_NAVIGATION_TIMEOUT_MS)` on
    the search box already tolerates it transparently, because Playwright
    locators poll for the element for the full timeout rather than
    checking once, so they naturally wait out the ~5s challenge. **Live
    reliability check: 5/5 consecutive runs of the real
    `amazon.search_product_page()` succeeded**, 4.2-6.2 seconds each,
    returning the correct FunFoods Mayonnaise product every time. The
    single reported `None` was very likely the same class of transient,
    probabilistic issue already documented for Zepto's login wall (e.g. a
    slower-than-usual WAF challenge, or a harder challenge tier under
    momentary load) - not a permanent site change.
    - **Fix applied, diagnostics only - behavior unchanged**: added
      `logger.debug`/`info`/`warning` calls throughout
      `search/retailers/amazon.py:search_product_page` (previously had NO
      internal logging at all, unlike `zepto.py`) - homepage navigation
      status, search-box/submit-button click failures (with page URL,
      title, and a body-text sample logged on timeout, matching a WAF
      challenge or a DOM change equally well without needing a second
      round-trip to re-diagnose), out-of-stock filter outcome, total
      result-card count, and which candidate index was chosen (with a
      sponsored-card-skip count). If `None` recurs, the logs will now show
      exactly which step failed instead of a bare `None`.

17. **[SUPERSEDED BY ITEM 18 - kept for history] First migration attempt:
    Qwen → Mistral Medium 3.5**, per explicit user request ("Qwen is no
    longer a viable option"). Root cause independently confirmed live: a
    direct call to `qwen/qwen3.5-122b-a10b` returned `HTTP 410: "The model
    ... has reached its end of life on 2026-07-20T00:00:00Z and is no
    longer available."` - this also retroactively explains the "HTTP 410"
    errors seen in earlier sessions' logs, which had been (incorrectly)
    assumed to be ordinary transient flakiness alongside HTTP 500s; they
    were not - the model was actually gone. `classifier/mistral.py`
    (`MistralVisionModel`, model `mistralai/mistral-medium-3.5-128b`) was
    implemented correctly per NVIDIA's own documentation, but **the model
    never returned a single response across two full sessions of live
    testing** (text-only, image, streaming, with/without
    `reasoning_effort`, timeouts up to 280 seconds, and even with a freshly
    rotated API key) - NVIDIA's own API reference for this model notes it
    runs under "NVIDIA API Trial Terms of Service" (trial-tier only),
    which likely explains the non-response. This entire attempt (the
    `classifier/mistral.py` file, `MistralVisionModel` class, all its call
    sites, `test_mistral.py`) was fully replaced by item 18 below - none
    of it remains in the codebase.

18. **Second migration attempt (this session) - Qwen → Nemotron 3 Nano
    Omni, succeeded and fully live-verified.** After Mistral's failure,
    the user asked for a **model-agnostic** search of NVIDIA Build's
    current catalog rather than picking another single model by name.
    Live-browsed build.nvidia.com/models (filtered toward Image-to-Text/
    VLM) and compared real candidates found there:
    - **nvidia/nemotron-3-nano-omni-30b-a3b-reasoning** (chosen): NVIDIA
      first-party, 33B (3B active MoE), explicitly tagged "OCR" and "VLM"
      on its own model card, has BOTH Free Endpoint AND Partner Endpoint
      availability (unlike Mistral's trial-only status - a materially
      stronger long-term-stability signal), 8M API calls in the last 30
      days at time of selection.
    - minimaxai/minimax-m3: 427B params, explicitly labeled "Preview" in
      its own description (weaker stability signal), no OCR tag.
    - moonshotai/kimi-k2: 1T params, Free Endpoint only (no Partner
      Endpoint), no OCR tag, much larger/costlier per call for no
      demonstrated benefit.
    Being NVIDIA's own model, not a third-party partner's, is the
    strongest long-term-availability signal available - exactly the kind
    of risk (a partner's business decision to retire a model) that caused
    Qwen's removal.
    - **What changed**: `classifier/mistral.py` → `classifier/nemotron.py`
      (renamed - class `MistralVisionModel` → `NemotronVisionModel`,
      provider key `"mistral_vl"` → `"nemotron_vl"`). Every call site
      updated: `retailer_gallery_stage.py`, `selection_stage.py`,
      `run.py`, `app.py`, `search/query_normalizer.py`'s docstring,
      `config.py`'s one comment naming the file. `test_mistral.py` →
      `test_nemotron.py` (and `test_downloader.py`'s one comment
      referencing it), same content otherwise.
    - **API differences, verified live and against NVIDIA's official
      model card rather than assumed**:
      - Endpoint and the image-embedding message shape
        (`{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<...>"}}`)
        are IDENTICAL to Qwen's/Mistral's. No message-building code needed
        structural changes.
      - Reasoning control is the SAME mechanism Qwen used:
        `"chat_template_kwargs": {"enable_thinking": false}` as a
        top-level request field - confirmed live. An `extra_body`-wrapped
        form (as shown in this model's *self-hosted* NIM container docs,
        a different doc set from the hosted Build API) was tried first
        and rejected by the hosted endpoint with a fast, clear
        `HTTP 400: "Unsupported parameter(s): extra_body"` - a much more
        diagnosable failure than Mistral's silent hangs, and a good
        illustration of why self-hosted-NIM docs and hosted-Build-API
        behavior can differ for the same model.
      - Max images per request: NVIDIA's docs did not state an explicit
        number for this model. Verified live instead: a request with 5
        real images (the exact count `select_best_candidate` sends per
        side) succeeded and the model correctly reported the count back -
        `_MAX_IMAGES_PER_REQUEST` kept at 5 on this empirical basis.
      - Verifier prompt (`prompts.PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT`)
        was **not** changed - nothing about this model's behavior required
        a prompt adjustment.
    - **Verification tests: COMPLETED, all passed.** Simple text request
      (0.6s, correct answer, reasoning cleanly separated into
      `reasoning_content`). Simple image request (2.8s, correctly
      identified brand/variant/on-pack tagline text - real OCR evidence).
      Production verifier prompt against real product images: a genuine
      match correctly accepted (`is_match=true, confidence=0.95`); a
      genuinely wrong product (Mayo image vs. Ketchup request) correctly
      rejected (`is_match=false, confidence=0.95`). **Full retailer
      pipeline run** (`run.build_stage()` → `stage.run("FUNFOODS MAYO
      COMBO 100g")`): Amazon search → gallery extraction → Nemotron
      verification (`is_match=true, confidence=0.95`) → full gallery
      downloaded and persisted to `images/funfoods_mayo_combo_100g/` with
      a correct `metadata.json`. Flipkart and Zepto were separately
      confirmed end-to-end too (search → gallery → Nemotron verification,
      each correctly accepted at confidence=0.95), directly proving all
      three retailers work with the new model.
    - **One genuine behavioral difference, reported honestly rather than
      hidden**: in one test, this model accepted a Tandoori Masala flavor
      variant against the generic ERP name "FUNFOODS MAYO COMBO 100g"
      (`is_match: true`), where Qwen previously rejected the identical
      case as a flavor mismatch. The ERP name does not explicitly specify
      a flavor, so this is a defensible edge-case judgment call, not a
      broken verifier - but it means flavor-unspecified generic ERP names
      may accept a named-flavor variant more readily than Qwen did. Worth
      watching for in production; not treated as a blocker since every
      unambiguous case (correct product accepted, wrong product rejected)
      behaved correctly.
    - **Pipeline wiring verified**: `run.build_stage()` constructs a full
      `RetailerGalleryStage` with `NemotronVisionModel` wired in (`.name`
      correctly returns `"nemotron_vl"`), and the `_RETAILERS` order
      remains `['Amazon', 'Flipkart', 'Zepto']`, unchanged. All imports
      across the whole codebase resolve correctly with no dangling
      references to `classifier/mistral.py`, `classifier/qwen.py`, or
      either retired class name.

19. **Fixed a brand-mismatch false-positive regression in the verifier
    prompt** (`prompts.py`, `PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT`
    only - no pipeline code touched, per explicit user request). User
    reported that Nemotron accepted ERP `"JK DHANIA POWDER 100 g PP"`
    against a real Amazon listing for `"Tata Sampann Coriander Powder
    100 g"` with `confidence=0.95`, reasoning that Dhania = Coriander and
    ignoring that "JK" and "Tata Sampann" are unrelated, clearly-legible
    brands. Root cause: the existing "different brand" rejection rule was
    a single, weakly-emphasized bullet with no worked example, while the
    synonym/house-brand acceptance guidance had a detailed, memorable
    worked example the model pattern-matched toward more strongly.
    - **Fix**: brand is now called out as an independently-judged
      identifier up front; a clearly-legible, genuinely different brand is
      now the FIRST and most heavily-emphasized rejection criterion, with
      an explicit anti-pattern warning ("a shared product category across
      brands is normal and expected, not evidence of a shared identity");
      a second worked example was added that mirrors the exact reported
      case (JK Dhania vs. Tata Sampann Coriander → must reject); and new
      confidence-calibration guidance makes a clear brand mismatch just as
      confidence-reducing as an unreadable attribute is confidence-
      preserving, while an genuinely illegible/ambiguous brand still falls
      back to evaluating the remaining identifiers instead of rejecting.
    - **Live-verified** against the real Amazon photo for "Tata Sampann
      Coriander Powder Masala" (`B079H39G18`), fetched fresh via
      `search.retailers.amazon` rather than assumed: (1) the exact
      reported regression - `JK DHANIA POWDER 100 g PP` vs. that photo -
      now correctly returns `is_match=false, confidence=0.95`; (2) the
      same photo against `TATA SAMPANN DHANIA POWDER 100g` (same brand,
      Dhania/Coriander synonym) still correctly returns
      `is_match=true, confidence=0.95`, confirming the synonym tolerance
      survived; (3) the original unreadable-weight accept case
      (`test_product_image2.jpg`, FunFoods Mayo) still returns
      `is_match=true, confidence=0.95`; (4) the original wrong-product
      reject case (same Mayo image vs. Kissan Ketchup) still returns
      `is_match=false, confidence=0.95`. All four cases behaved as
      required - no new false negatives introduced by the tightened
      brand-mismatch language.
    - **Not touched by this fix**: the pre-existing Tandoori-Masala-
      flavor-variant behavioral difference (item 18 / Known Bugs) -
      re-tested against the same image (`test_product_image.jpg`) and it
      is still accepted (`is_match=true, confidence=0.95`) against the
      flavor-unspecified generic ERP name `FUNFOODS MAYO COMBO 100g`. This
      is a distinct edge case (flavor ambiguity, not brand identity) that
      the user did not ask to be addressed this time, so it was
      deliberately left as-is and remains an open, known item.

---

# Known Bugs / Remaining Issues

- **One genuine behavioral difference between Nemotron 3 Nano Omni and
  Qwen** (the current verifier model - see item 18): in one live test,
  Nemotron accepted a Tandoori Masala flavor variant against the generic
  ERP name "FUNFOODS MAYO COMBO 100g" (`is_match: true`), where Qwen
  previously rejected the identical case as a flavor mismatch. The ERP
  name doesn't explicitly specify a flavor, so this is a defensible
  edge-case judgment call, not a broken verifier - but worth watching for
  in production if flavor-unspecified generic ERP names start accepting
  named-flavor variants more often than expected. Unambiguous cases
  (correct product accepted, wrong product rejected) both behaved
  correctly, and the unreadable-weight-should-not-reject behavior was
  specifically re-confirmed on the exact same real image used for the
  original Qwen fix.
- **Zepto occasionally gates search results behind a login wall** ("Please
  login to continue searching") instead of showing results - confirmed
  live (see item 14 in Problems Already Solved) to be transient/
  probabilistic (immediate retries of the identical query, and of the
  identical Amazon->Flipkart->Zepto sequence, both succeeded), not tied to
  a specific query or to running after other retailers. `zepto.py` now
  detects and logs this distinctly from a genuine empty result, but does
  NOT retry on it - when it happens, `search_product_page` returns `None`
  and the pipeline falls through to the next retailer/Google Images for
  that run, same as any other miss. A retry-with-fresh-context specifically
  on this signal was considered but not implemented (out of scope for a
  root-cause-only task) - worth doing if this proves to happen often
  enough to matter in practice.
- **Godrej Jersey Kesar Flavoured Milk is confirmed unreachable via
  Zepto's own search** (see item 15) - the product exists and is
  purchasable at its direct URL, but does not appear in Zepto's results
  for any of 5 broader queries tested (spanning brand, category, and
  flavour), even ones that surface many sibling products from the same
  brand/category. This is treated as a genuine miss (falls through to the
  next retailer), not a bug - there is no known query that retrieves it
  through Zepto's own search.
- **Amazon.in fronts every request with AWS WAF bot mitigation** (see item
  16) - a normal, ~5-second, self-resolving challenge (HTTP 202 -> empty
  body -> `challenge.js`/`inputs`/`mp_verify` -> HTTP 200 with real
  content), not a CAPTCHA or login wall. Currently tolerated transparently
  by Playwright's auto-retrying `.click()`, confirmed reliable across 5
  consecutive live runs. `amazon.py` now logs each step so if this WAF
  challenge is ever slower than the `_NAVIGATION_TIMEOUT_MS` (20s) budget,
  or Amazon changes the challenge/DOM, the exact failing step is visible
  in logs instead of a bare `None`.
- **Blinkit and BigBasket were removed entirely** (an earlier session, by
  explicit user request, to stabilize on Amazon + Flipkart before adding
  more retailers back) - so their prior sandbox-specific Playwright
  blocking issues (below, kept for history) are no longer relevant to the
  active pipeline. Re-adding either later would need fresh selector
  verification from scratch, per this project's "never guess selectors"
  rule, since their extractor modules no longer exist. **Zepto was
  re-investigated and re-added this session** (see "Zepto re-added as a
  retailer..." in Problems Already Solved) - it is no longer in this
  removed-and-unconfirmed category; its search/extraction were confirmed
  live end-to-end from Python/Playwright in this sandbox, not just via
  manual browser-tool DOM inspection.
  Blinkit's blocking issue is very likely an environment-specific
  (network/IP-reputation or browser-fingerprint) issue, not necessarily
  present on a normal residential network.
- **Query-normalization end-to-end test was interrupted** (the user pivoted
  mid-verification to the retailer-native-search discussion). The
  `normalize_for_search()` function itself is thoroughly unit-tested and
  correct (verified with many examples). Its WIRING into
  `retailer_gallery_stage.py` for the Zepto/Blinkit SerpAPI path was
  carefully re-traced by hand after the native-search refactor and should
  be correct, but has not been re-confirmed with a fresh live/mocked
  end-to-end run since that refactor landed.
- **No automated test coverage** for: `retailer_gallery_stage.py`,
  `search/retailers/*.py`, `search/query_normalizer.py`,
  `search/retailer_web_search.py`, `ProductMatchVerdict`. All verification
  so far has been via ad-hoc scratch scripts (in the OS temp scratchpad
  directory, not committed anywhere) and manual live runs — real and
  thorough, but not regression-proof or repeatable via `pytest`.
  `tests/test_models.py` (6 tests) only covers the original domain models.
- **RESOLVED - debug print statements in `classifier/nemotron.py` converted
  to proper logging** (carried unchanged through the Qwen→Mistral→Nemotron
  migrations, then finally addressed in the post-review cleanup - see the
  matching entry earlier in this section). `_debug_print_request` became
  `_log_request_debug` (a guarded `logger.debug` call); the two
  `_extract_message_content` prints became `logger.error` calls. `os` is
  still imported/needed for `_BASE_URL_ENV_VAR` lookup, unaffected by this.
- **Flipkart ad/sponsored-result exclusion was not found necessary** in the
  one live test performed (no "Ad" labels appeared for that query), but
  this was NOT exhaustively verified across many queries the way Amazon's
  sponsored-filtering was. If Flipkart does inject ads for other queries,
  `flipkart.py:search_product_candidates` does not currently filter them out.
- **`front_image_path`/`back_image_path`/`front_confidence`/`back_confidence`
  are always `None` for retailer-gallery-mode results** — this is a
  deliberate design choice (see Important Design Decisions), not a bug,
  but it means any future code reading a `ProductImageResult` must check
  `metadata.get("retailer")` to know which "mode" produced it before
  knowing where to look for the actual image paths.
- **RESOLVED - legacy/dead code and temporary test scripts removed** (a
  later session, as part of a full repo cleanup before preparing this
  project for GitHub). Confirmed via grep - zero real imports anywhere,
  only self-references and stale doc-comment mentions - before deleting:
  `main.py`, `pipeline.py`, `selector/` (the whole package -
  `confidence_selector.py` was its only real content, and its interface,
  `interfaces.ImageSelector`, was removed too since it had no remaining
  implementation), `search/google_images.py`, `search/google_search.py`,
  and the five temporary manual test scripts (`test_google.py`,
  `test_serpapi.py`, `test_downloader.py`, `test_nemotron.py`,
  `test_selection_stage.py` - each was explicitly self-described in its own
  docstring as "not part of the project's test suite... safe to delete").
  Also removed `output.txt`, a stray captured-stdout artifact from an early
  Qwen-era manual test run. Stale doc-comment references to the deleted
  files (in `search/__init__.py`, `search/serpapi.py`, `selection_stage.py`,
  `search/query_generator.py`, `interfaces.py`) were updated to describe the
  current architecture instead. `pyproject.toml`'s dependency list was also
  found stale at the same time (missing `playwright` and `pillow`, both
  real, active dependencies) and corrected. The real, structured test suite
  (`tests/test_models.py`) was untouched - it's the one explicitly
  referenced by every deleted script's "not part of the project's test
  suite (see tests/)" docstring line.
- **RESOLVED - remaining dead code from the pre-retailer-first skeleton
  removed, following a full architectural review** (grep-confirmed zero
  real callers before touching anything): `models.RetrievalRequest`
  (constructed nowhere except its own now-removed unit test);
  `config.Settings`' `max_queries`, `max_candidates_per_query`,
  `min_confidence`, `max_concurrent_downloads`,
  `max_concurrent_classifications`, `enabled_search_providers`,
  `active_vision_provider`, `active_storage_provider`, `log_level` (none
  read anywhere outside `config.py` itself - `run.py`/`app.py` both call
  `configure_logging("INFO")` with a hardcoded literal, never
  `settings.log_level`), plus the `SearchProviderKey`/`VisionProviderKey`/
  `StorageProviderKey` enums that backed the three provider-selection
  fields. `classifier/nemotron.py`'s `classify_image()`/`classify()` (plus
  their private `_build_messages`/`_parse_response`/`_EXTRA_FIELDS_INSTRUCTIONS`
  support) were also removed - neither real orchestrator ever called them,
  both use `select_best_candidate()`/`verify_product_match()` instead.
  Removing `classify()` meant `NemotronVisionModel` no longer satisfies
  `interfaces.VisionModel`'s abstract contract, so it was changed to a
  plain class (no longer inheriting `VisionModel`) rather than left
  half-implementing an ABC - confirmed nothing anywhere types against
  `VisionModel` polymorphically, so this loses no real capability. The 10
  remaining `print("[DEBUG] ...")` statements in `classifier/nemotron.py`
  (self-labeled `TEMPORARY` since an earlier debugging phase, tracked as a
  known issue for a long time - see the "Debug print statements" bug entry
  below, now itself resolved) were converted to `logger.debug`/
  `logger.error` calls - the request-dump one is now guarded by
  `logger.isEnabledFor(logging.DEBUG)` so it's silent by default and the
  deep-copy/truncation work it does isn't wasted when nobody's watching.
  Live-verified after all of the above: `NemotronVisionModel()` still
  instantiates and works end-to-end (Amazon accept, confidence 0.95, zero
  `[DEBUG]` console output at INFO level). Also added `.github/workflows/ci.yml`
  running ruff + pytest + mypy on push/PR - mypy is `continue-on-error`
  for now since it reports 17 pre-existing strict-mode findings unrelated
  to this change (missing `openpyxl` stubs, a few `Any`-returns, some
  untyped Streamlit callback parameters) that weren't in scope to fix here.
- **RESOLVED - `interfaces.VisionModel` removed following a final design
  review of `interfaces.py`.** Evidence gathered before touching anything:
  grepped inheritance (`class X(VisionModel)`) and every other reference
  across the whole repo - zero classes implemented it (the previous entry
  already removed its one implementer, `NemotronVisionModel`) and zero
  functions/constructors ever annotated a parameter as `VisionModel`. This
  wasn't a recent regression either - across all three real vision-model
  migrations this project has done (Qwen -> Mistral -> Nemotron), every
  swap was done by changing concrete imports/constructors directly; the
  port never once mediated an actual swap, and its one abstract method
  (`classify`) doesn't even match what real callers need
  (`select_best_candidate`/`verify_product_match`). The other four ports in
  `interfaces.py` (`QueryGenerator`, `ImageSearchProvider`, `ImageDownloader`,
  `StorageProvider`) were reviewed the same way and kept - each has exactly
  one concrete implementation AND at least one real caller that
  type-annotates against the abstract port (not the concrete class),
  confirmed via the same inheritance+annotation grep. `interfaces.py`'s
  module docstring now documents this evidence directly (a small table of
  port -> implementation -> annotated-in) so the next reviewer doesn't have
  to re-derive it. Stale doc-comment mentions of `interfaces.VisionModel` in
  `classifier/nemotron.py`, `classifier/__init__.py`, and `selection_stage.py`
  were updated to match. Also added `docs/mypy-remediation.md`: the 17
  mypy findings grouped into 5 fix-once-per-group categories (missing
  `openpyxl` stubs, `str`-vs-`HttpUrl` mismatches, `Returning Any`,
  missing annotations, one Playwright `ViewportSize` mismatch), each with
  a concrete fix approach and a suggested order, so mypy's CI step can go
  from `continue-on-error` to a real merge gate incrementally rather than
  in one large, riskier pass. Verified after removing `VisionModel`: ruff
  clean, `pytest tests/` 5/5, all 34 modules import cleanly, mypy still
  reports exactly 17 errors (none added/removed by this change), and a
  fresh live end-to-end run still accepts on Amazon at confidence 0.95.

---

# Future Improvements

Ideas discussed but intentionally postponed / not yet started:

- **Semantic search / embeddings** for product matching — not discussed in
  depth, no decision made either way; current approach is rule-based
  normalization (search) + VLM judgment (verification), no embedding
  similarity anywhere in the pipeline.
- **Better retailer ranking** — currently a fixed priority order
  (Amazon → Flipkart) tried sequentially until one accepts; no dynamic/
  learned ranking, no parallel racing of retailers.
- **UI** — see "UI status" under Components Implemented; a Streamlit UI
  (`app.py`) exists with single-product and Excel-batch tabs. This bullet
  is stale/superseded from an earlier point in the project and kept only
  for history.
- **Re-adding Blinkit/BigBasket** — removed entirely in an earlier session
  by explicit user request. Zepto is DONE (re-added with fully native
  search - see item 13 in Problems Already Solved); this bullet now only
  applies to Blinkit/BigBasket. Their old extractor modules no longer
  exist; re-adding either later would need fresh live selector
  verification from scratch (same rigor as Amazon/Flipkart/Zepto),
  including re-diagnosing whether Blinkit's earlier "blocked" finding was
  also just AWS-WAF-style bot mitigation rather than a fundamental
  limitation, the way Zepto's turned out to be.
- **Sponsored/ad exclusion for Flipkart's native search** — not yet
  verified as necessary, but not yet ruled out either; would need testing
  across a wider variety of queries.
- **Cleaning up temporary debug prints** in `classifier/nemotron.py` —
  flagged as remaining but not yet actioned.
- **Re-enabling temp-file cleanup** (`_cleanup_temp_files` in
  `selection_stage.py`) — currently disabled for development-mode
  inspection; would need to be revisited for a "production" deployment
  mode where disk usage matters.
- **Formal automated test coverage** for all the new retailer-first
  components — see Known Bugs / Testing Status.
- **Verification prompt further tuning** — the current prompt handles the
  cases discussed so far well (including a live-confirmed nuanced case:
  correctly declining to confirm weight when it's not visible in the
  photo), but has not been stress-tested across a large, diverse set of
  real ERP names/retailer pairs.

---

# Testing Status

**What has been tested (real, live verification, not just unit tests):**
- `search/retailers/amazon.py`: both `extract_gallery` and
  `search_product_page` — real Playwright runs from this sandbox, multiple
  queries, correct results including sponsored-filtering.
- `search/retailers/flipkart.py`: both functions — real Playwright runs,
  including discovering and fixing the login-popup and duplicate-input
  bugs live.
- `search/retailers/zepto.py`, `blinkit.py`: `extract_gallery` verified via
  interactive browser-tool DOM inspection with concrete evidence
  (byte-size comparisons); native search NOT attempted for these two;
  `extract_gallery` NOT confirmed to work via bare Playwright in this
  sandbox specifically (blocked, see Known Bugs) — though the extraction
  LOGIC itself was verified against real, live-rendered DOM.
- `search/query_normalizer.py`: thoroughly unit-tested inline (many
  examples: FUNFOODS MAYO COMBO, KISSAN...PCH, MILKY MIST...TIN, GEMINI
  DUST TEA, edge cases like empty string, already-canonical units, mixed
  case) — all passed. Not re-verified end-to-end after the native-search
  refactor (see Known Bugs).
- `classifier/nemotron.py::verify_product_match` (formerly
  `classifier/qwen.py`, migrated this session via a failed Mistral
  attempt - see items 17-18): the Qwen-era implementation was unit-tested
  (message building, response parsing, accept/reject cases) AND confirmed
  against the real live NVIDIA Build API in an earlier session's final
  test. **The Nemotron version is also now live-verified**: real matching
  product accepted, real non-matching product rejected, full retailer
  pipeline run (Amazon) accepted and persisted, Flipkart and Zepto
  separately confirmed - see item 18.
- `classifier/nemotron.py::select_best_candidate`, `classify_image`:
  the Qwen-era implementation was verified in earlier turns (unit +
  mocked-server end-to-end). For Nemotron, `select_best_candidate`'s
  5-image support was live-verified (see item 18); `classify_image`
  itself has not been separately re-run this session but shares the same
  underlying request machinery already proven working by the other two
  entry points.
- `download/downloader.py`, `storage/local_filesystem.py`
  (`persist_gallery` included): unit + integration tested earlier this
  session, including a real local HTTP server test covering success,
  corrupt image, 404, invalid URL, and duplicate-URL cases.
- `retailer_gallery_stage.py` (the whole orchestrator): tested end-to-end
  with fake/mocked retailer configs (reject → next retailer → accept →
  persist; all-reject → fallback called); in an earlier session, with ONE
  fully real run (real Amazon search + real gallery + real download +
  real Qwen call) that correctly exercised the full path down to a real
  (conservative) rejection and fallback trigger; and, this session, **a
  full real accept-and-persist run against the new Nemotron verifier**
  (Amazon search → gallery → verification → full gallery persisted +
  metadata.json written for "FUNFOODS MAYO COMBO 100g" - see item 18).
- Existing `tests/test_models.py` (6 tests): still passing, unaffected by
  any of this session's changes.

**What still needs testing:**
- A full real run where a retailer is actually ACCEPTED end-to-end (the
  one real full-pipeline test performed ended in a legitimate rejection
  due to weight not being visible — never got to see the "download full
  gallery + persist_gallery" path exercised against the REAL API in one
  single run, though `persist_gallery` itself was separately unit-tested
  and the accept-path logic was covered by the mocked orchestrator test).
- Blinkit end-to-end from a non-sandboxed network (to determine if its
  blocker is truly environment-specific). Zepto's search + extraction were
  confirmed live end-to-end this session (see Problems Already Solved,
  item 12) and no longer need this.
- Zepto specifically still needs a real full-pipeline run reaching it as
  the ACCEPTED retailer (both live tests this session called
  `search_product_page`/`extract_gallery` directly, not through a full
  `RetailerGalleryStage.run()` where Amazon/Flipkart are tried first).
- Formal `pytest`-based regression tests for anything under `search/`
  (beyond `query_generator.py`, which isn't separately tested either) and
  for `retailer_gallery_stage.py`.
- Flipkart ad-exclusion across a wider variety of queries.

---

# Current TODO

**Both of the previously-blocking items are now resolved as of this
session**: the vision-model migration (Qwen → Nemotron 3 Nano Omni, after
a failed Mistral attempt - see items 17-18 in Problems Already Solved) is
fully live-verified, AND a fully real, fully successful end-to-end
accept-and-persist run was completed as part of that verification (real
Amazon search → real gallery extraction → real Nemotron acceptance → real
`persist_gallery` + `metadata.json` write for "FUNFOODS MAYO COMBO 100g",
plus separate confirmation that Flipkart and Zepto also search/extract/
verify correctly with the new model). There is no longer an open,
blocking gap in live verification of the core retailer-first path.

Remaining, in rough priority order:
1. Watch for the one known behavioral difference from Nemotron (accepting
   a named-flavor variant against a flavor-unspecified generic ERP name -
   see item 18 and Known Bugs) in real production use, and reconsider the
   verifier prompt if it proves to matter in practice.
2. Decide whether to invest in formal `pytest` coverage for the
   retailer-first components (currently zero automated regression tests
   for anything built this session).
3. Decide whether/when to clean up the temporary debug prints in
   `classifier/nemotron.py`.
4. Decide whether to revisit Blinkit (currently removed) now that Zepto's
   fully native search pattern is proven, or leave it out permanently.
   Zepto itself is done - re-added and live-verified this session, with
   native search (not SerpAPI - that approach was tried and replaced, see
   items 12-13).
5. **RESOLVED** - the legacy/unused files and temporary test scripts were
   deleted in the GitHub-prep cleanup (see "RESOLVED - legacy/dead code..."
   in Known Bugs / Remaining Issues above).

---

# Conversation Memory

Assumptions, preferences, constraints, and decisions that should carry
into the next chat:

- **User's core workflow context**: product names entered into this system
  are **ERP/internal inventory names**, not consumer-facing marketing
  titles. They routinely differ from retailer listings in wording,
  abbreviation, word order, and included/omitted marketing language. This
  is the central problem the whole retailer-first + normalization +
  tolerant-verification design exists to solve.
- **"Do not guess selectors" is a hard, repeatedly-enforced rule** in this
  project. Every retailer-specific selector/JSON-path claim in the code was
  derived from live inspection (via the interactive browser tool and/or
  real Playwright runs) with evidence cited in comments — never assumed
  from general knowledge of "how these sites probably work." This
  discipline should continue for any new retailer or site touched in the
  future.
- **User consistently wants trade-offs explained BEFORE implementation**
  for any significant architectural change (seen twice this session: the
  retailer-first redesign, and the native-search-vs-SerpAPI decision). Both
  times, a clear written analysis was given, often followed by an
  `AskUserQuestion` to let the user pick between concretely-described
  options, rather than the assistant unilaterally deciding. Continue this
  pattern for future major changes.
- **User values honesty about environment-specific limitations** over
  papering over them — multiple times this session, a "doesn't work"
  finding was investigated all the way to its true root cause (bot
  detection vs. selector bug vs. genuine site change) and reported
  precisely, rather than assumed. Continue distinguishing "this sandbox's
  network is blocked" from "this code is wrong."
- **New capabilities are added as new, additive, concrete methods** —
  never by modifying/removing an existing method that something else might
  depend on, and never by adding to `interfaces.py` unless multiple
  concrete implementations actually need the shared contract. This pattern
  should continue.
- **SUPERSEDED this session**: "development mode" (downloaded candidates
  NOT deleted, kept for inspection) is **no longer the case**. The user
  explicitly chose (via `AskUserQuestion`) to move ALL candidate downloads —
  for both the retailer-first stage and the Google-Images fallback — into
  ephemeral OS temp directories that are always cleaned up, in order to get
  a single-root `images/` layout with no scratch/output split. Only
  final, accepted/selected images now persist to disk. Do not assume
  rejected candidates are recoverable for inspection going forward.
- **The user prefers minimal-footprint changes** — e.g. reusing
  `ProductImageResult.metadata` (a free-form dict) for new
  gallery-mode-specific fields rather than adding new top-level model
  fields, to avoid touching shared/core files (`models.py`) more than
  necessary. `ProductMatchVerdict` was the one explicit exception, added as
  a genuinely new top-level model because the user specifically asked for
  it (separate responsibility from `ClassificationVerdict`).
  Interfaces (`interfaces.py`), `models.py`, and `config.py` are treated as
  "core/shared" files to be touched carefully and only when clearly
  justified.
- **Live/real verification is strongly preferred over trusting mocks
  alone** wherever feasible — this session repeatedly followed up mocked
  end-to-end tests with at least one fully real run (real API keys, real
  websites) before considering a feature "done," and was explicit and
  honest on the rare occasions a fully-real run wasn't performed (e.g.
  Zepto/Blinkit's automated extraction has only been verified via the
  interactive browser tool, never via a real unattended Python/Playwright
  run in this sandbox — this is flagged, not glossed over).
- **Development environment**: Windows, PowerShell as the primary shell but
  a Git Bash tool also available and used throughout this session;
  `uv`-style dependency management is encouraged for this project, though
  `pip`/system Python was used directly for ad-hoc verification throughout
  this session.
- **API keys are real and live** (`.env` file with genuine
  `NVIDIA_API_KEY` and `PIR_SERPAPI_API_KEY` values) — this project is not
  running against pure mocks; real API calls (with real cost/quota
  implications) have been made throughout, and should continue to be used
  judiciously (mock-first for iterative debugging, real calls for final
  confirmation) rather than avoided entirely or overused.
- **Retailer priority order has changed multiple times over the
  conversation** — Amazon → Flipkart → Zepto → Blinkit was the order for
  several sessions, but Zepto/Blinkit/BigBasket were REMOVED ENTIRELY in
  the most recent session, by explicit user request (to stabilize on two
  retailers before adding more back). The CURRENT, final, confirmed order
  is simply **Amazon → Flipkart → Google Images (fallback)**. Do not
  re-add Zepto/Blinkit/BigBasket/JioMart without the user explicitly
  asking again - and if asked, their old code is gone, so treat it as
  building a new provider from scratch (fresh live selector verification
  required).
- **[OUTDATED - see items 17-18 in Problems Already Solved] Model name has
  changed** — the vision model is no longer Qwen at all.
  `qwen/qwen3.5-122b-a10b` (itself a change from the originally-used
  `qwen/qwen3.5-397b-a17b`) reached end-of-life on 2026-07-20 and returns
  HTTP 410 from NVIDIA Build. A first migration attempt to
  `mistralai/mistral-medium-3.5-128b` never worked (the model never
  responded on NVIDIA Build's trial-tier endpoint) and was fully removed.
  The verifier is now `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
  (`classifier/nemotron.py`, `NemotronVisionModel`), chosen via a
  model-agnostic catalog search and fully live-verified. Do not revert to
  Qwen (gone) or Mistral Medium 3.5 (never worked in this sandbox).
