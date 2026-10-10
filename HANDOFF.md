# Handoff — Shop & Compare

Current-state engineering notes for anyone picking this project up. For setup and usage see
[README.md](README.md). The original chronological log (written before the project became
Shop & Compare, and describing several behaviors that no longer exist) is archived in
[docs/archive/HANDOFF-history.md](docs/archive/HANDOFF-history.md); read it only for the rationale
behind old decisions.

## 1. What the project is

A Streamlit app (`app.py`) with two tabs:

- **Shop & Compare** (the main feature): one product name in, a price-sorted list of offer cards out.
- **Single product**: retrieves verified product images for one product name (older feature, still
  supported, also available as a CLI via `run.py`).

Everything is offline-testable: the test suite fakes every provider and a network guard
(`tests/conftest.py`) fails any test that tries to reach a non-local host.

## 2. Shop & Compare pipeline

Orchestrated by `ShoppingComparisonStage.compare()` in `shopping_comparison_stage.py`:

```
SerpApi Google Shopping search (1 call)
  -> approved-retailer filtering          shopping_retailers.py
  -> first 10 approved candidates         _MAX_CANDIDATES
  -> retailer-URL resolution              google_immersive_product, 1 call per candidate
  -> product-page text evidence           product_page_text.py (700-char cap)
  -> Nemotron classification              classifier/nemotron.py, prompts.py
  -> Python ranking                       rank_comparison_offers()
  -> offer cards                          app.py
```

| Step | Details |
|---|---|
| Search | `search/retailers/google_shopping_serpapi.py` → `search_shopping_offers`. One fixed page (~40 results); `_MAX_RAW_SHOPPING_RESULTS = 100` simply reads all of it. The query goes through `search/query_normalizer.py` (strips ERP packaging codes, fixes unit spacing). |
| Filtering | `canonical_retailer()` does **exact** matching of the normalized seller name against an alias table. Never substring matching. Unknown sellers are discarded with no fallback. |
| Candidate cap | `_MAX_CANDIDATES = 10`, taken in Google's order. This bounds SerpApi usage (1 + 10 calls) and NVIDIA load. |
| URL resolution | `resolve_direct_merchant_link()` calls `google_immersive_product` for each candidate. The URL resolved here is the one the UI links to; it is never resolved twice. |
| Evidence | Title, retailer, price and URL, plus bounded text from the retailer page (`fetch_product_page_text`, never raises, returns `None` on any failure). The thumbnail is supplemental. Nothing is required: a missing URL, page or thumbnail just means less evidence. |
| Classification | Batches of at most `_MAX_OFFERS_PER_CLASSIFICATION_BATCH = 5`, run concurrently; each failed batch is tried `_BATCH_ATTEMPTS = 3` times with 5 s then 10 s pauses. Response limit `_SHOPPING_CLASSIFICATION_MAX_TOKENS = 800`. Malformed JSON is salvaged by a tolerant parser; unknown labels are mapped by `_SHOPPING_LABEL_ALIASES` (for example `not_relevant` → `different_variant`). |
| Ranking | Classification order (same product, same product in another size, related), then cheapest, then Google's position; only the cheapest duplicate per retailer and classification is kept. `different_variant` offers are excluded from the ranked list. |
| Display | *Comparable offers* (same product, same product different size), *Related alternatives*, and a caption counting hidden unrelated listings and offers that "couldn't be verified". |

### Classification semantics (`models.ShoppingMatchClassification`)

`same_product`, `same_product_different_size`, `related_product`, `different_variant`. Nemotron judges
**product identity only**; ordering, grouping and price handling are Python's job. The prompt rules
worth knowing about: brand ownership (a sub-brand of the requested brand can match), a container,
empty packaging or accessory listing fails closed to `different_variant`, and size differences are
only "same product" when the product itself is identical.

### Failure handling principle

A classification failure is an infrastructure failure, not evidence of a mismatch. Offers whose
batch could not be classified are kept in `ShoppingComparisonResult.unclassified_offers` and
reported as a count; they are never shown as results and never treated as matches.

## 3. Single product pipeline

`run.build_stage()` wires `RetailerGalleryStage` (`retailer_gallery_stage.py`) with
`ProductImageSelectionStage` (`selection_stage.py`) as its fallback.

1. Normalize the search query (verification always sees the original name).
2. Try retailers in order: **Amazon** (`engine=amazon` / `amazon_product`), then **Google Shopping**
   (`google_shopping_serpapi.search_product_page` / `extract_gallery`). Up to 5 candidates each.
3. Download only the first gallery image to a temp directory, ask Nemotron
   (`verify_product_match`) whether it is the requested product, then apply
   `verification_overrides.py` (a deterministic, one-directional brand-mismatch guard that can only
   turn an accept into a reject).
4. On acceptance, download and persist the full gallery to `images/<product_slug>/image_NN.<ext>`
   with a `metadata.json`.
5. If both retailers fail and `ENABLE_GOOGLE_FALLBACK` is true (the default), run the Google Images
   comparative selection (`selection_stage.py`, `search/serpapi.py`), producing `front` / `back`
   images.

No browser is launched anywhere in this path (`page` is passed as `None`).

## 4. Module map

| Module | Role |
|---|---|
| `app.py` | Streamlit UI. Native Streamlit components only, no custom CSS. `main()` sets the page config (title `Shop & Compare`), the heading and tagline, and the two tabs. `_render_offer_row()` draws one card. |
| `run.py` | CLI plus the composition roots `build_stage()` (single product) and `build_shopping_stage()` (Shop & Compare). |
| `shopping_comparison_stage.py` | Shop & Compare orchestration, batching, retries, reconciliation, ranking. |
| `shopping_retailers.py` | `APPROVED_RETAILERS`, `normalize_source`, `canonical_retailer`. |
| `product_page_text.py` | Standard-library HTML text extraction with an 8 s fetch timeout and a 700-character cap. |
| `classifier/nemotron.py` | NVIDIA Build client: `verify_product_match`, `select_best_candidate`, `classify_shopping_offers`. Model `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, thinking disabled, at most 5 images per request. |
| `prompts.py` | All prompts. |
| `models.py` | Pydantic models (`ShoppingOffer`, `ShoppingMatchVerdict`, `ClassifiedShoppingOffer`, `ShoppingComparisonResult`, `ProductImageResult`, ...). |
| `config.py` | `pydantic-settings` `Settings`; env prefix `PIR_`, plus the bare `NVIDIA_API_KEY`. The only place that reads the environment. |
| `search/` | Query normalization/generation, SerpApi Google Images, and `retailers/` (Amazon and Google Shopping via SerpApi). |
| `download/`, `storage/`, `utils/` | Image downloader, local filesystem persistence, logging/slug/hash helpers. |

## 5. Things to leave alone unless you mean to change behavior

These were tuned against live runs; each has tests that pin it.

- The approved-retailer allowlist and its exact-match rule.
- The Nemotron prompts, the label alias table, the JSON salvage parser and the token limits.
- `_MAX_CANDIDATES`, the batch size and the retry policy.
- Merchant-link resolution before classification, and the product-page text extraction limits.
- The deterministic ranking and duplicate handling.

To add a retailer, add its already-normalized aliases to `APPROVED_RETAILERS` and a test in
`tests/test_shopping_retailers.py`.

## 6. Configuration

See `.env.example`. `NVIDIA_API_KEY` and `PIR_SERPAPI_API_KEY` are required;
`PIR_REQUEST_TIMEOUT_SECONDS` is optional (default 15, the NVIDIA client enforces a 60 s minimum for
its own requests). `.env` is git-ignored.

## 7. Testing and tooling

```powershell
python -m pytest
python -m ruff check .
python -m mypy .
```

- Pytest and Ruff are clean; both are blocking in CI (`.github/workflows/ci.yml`).
- The Ruff rule set is pinned in `pyproject.toml` so newer Ruff releases cannot change results.
- Mypy (`strict = true`) still reports findings and is non-blocking in CI; see
  [docs/mypy-remediation.md](docs/mypy-remediation.md).
- UI tests use a recording stand-in for `st` (`tests/test_shop_compare_ui.py`), so layout structure
  is asserted without starting Streamlit.
- Live behavior (real SerpApi, NVIDIA and retailer pages) is validated manually and spends credits;
  it is deliberately not part of the automated suite.

## 8. Known limitations

- Google Shopping results vary between runs; only the first 10 approved candidates are examined.
- NVIDIA Build rate limits and transient errors still occur. Retries reduce, but do not eliminate,
  "couldn't be verified" offers.
- Brand-only or generic queries can be classified inconsistently.
- Some retailers block automated page fetches or render content client-side, so page text is
  sometimes unavailable.
- Prices are those shown by Google Shopping and are not re-checked on the retailer site.
- Searches are India-specific (`gl=in`, `google.co.in`, Indian retailers).

## 9. Legacy and dormant code

Kept deliberately, not wired into the running app:

- `batch/` — Excel import, per-row orchestration and ZIP packaging for a Batch (Excel) tab that was
  removed from the UI. Nothing imports it. It is the only user of the `openpyxl` dependency and has
  no tests. Safe to delete together with `openpyxl` if it is not wanted.
- `search/retailers/amazon.py`, `flipkart.py`, `zepto.py` — older Playwright-based retailer
  scrapers, left in place as a rollback. Nothing in the active pipeline imports them. They (and the
  type imports in a few active modules) are why `playwright` is still a dependency; no browser
  binary is needed at runtime.

## 10. Repository notes

- Licensed under MIT (see `LICENSE`).
- `images/` (generated output) and `.env` are git-ignored.
- `.claude/launch.json` defines the Streamlit dev-server entry; `.claude/settings.local.json` is
  git-ignored.
