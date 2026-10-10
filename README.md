# Shop & Compare

**Compare prices from trusted retailers. Find the right product, faster.**

Shop & Compare is a Streamlit app that takes a product name, finds listings for it on Google
Shopping, keeps only approved Indian retailers, and uses an LLM (NVIDIA Nemotron) to decide which
listings are really the product you asked for. Deterministic Python code then ranks them by
relevance and price and shows each one as an offer card with the retailer, a large price and a
**View offer** button.

The app has two tabs:

| Tab | What it does |
|---|---|
| **Shop & Compare** | Price comparison across approved retailers (the main feature) |
| **Single product** | Retrieves verified, high-resolution product images for one product name |

## Screenshots

No screenshots are included yet. Add real ones under `docs/screenshots/` and link them here.

## Key features

- **Approved retailers only.** Listings from any seller outside a fixed allowlist
  (`shopping_retailers.py`) are discarded, with exact name matching so that "Shopsy By Flipkart"
  is not mistaken for Flipkart.
- **Evidence-based matching.** For each candidate the app resolves the retailer's own product URL,
  reads a bounded amount of text from that page and sends it, with the listing title, price and
  thumbnail, to Nemotron. Text is the primary evidence; the thumbnail is supplemental.
- **Four-way classification.** Every candidate is labelled `same_product`,
  `same_product_different_size`, `related_product` or `different_variant`. Nemotron judges the
  product; it never sorts or picks a winner.
- **Deterministic ranking.** Python orders offers by classification, then cheapest first, then
  Google's original position, and keeps only the cheapest duplicate per retailer.
- **Honest empty and failure states.** Different products are hidden but counted. Offers that could
  not be classified (for example after an API failure) are reported as "couldn't be verified"
  instead of being shown or silently treated as matches.
- **Responsive, price-focused cards** built from native Streamlit components only.

## How it works

```
SerpApi (Google Shopping search)
  -> approved-retailer filtering
  -> product-page evidence (direct retailer URL + page text)
  -> Nemotron classification
  -> Python ranking
  -> offer cards
```

1. **Search.** One Google Shopping search through SerpApi (`engine=google_shopping`, India).
2. **Filter.** Keep only listings from approved retailers, then take the first 10 in Google's order.
3. **Evidence.** For each of those, one `google_immersive_product` call resolves the direct
   retailer URL; the retailer page is fetched and a short text excerpt is extracted.
4. **Classify.** Candidates go to Nemotron in batches of up to 5 (with retries on transient NVIDIA
   errors). Each gets a classification, a confidence and a reason.
5. **Rank and display.** `rank_comparison_offers` produces the final order. The UI shows
   *Comparable offers* (same product, or same product in a different size) and
   *Related alternatives*.

SerpApi usage per comparison is 1 search plus up to 10 `google_immersive_product` calls.

The **Single product** tab runs a separate pipeline: Amazon (via SerpApi), then Google Shopping,
then a Google Images fallback. A vision model verifies each candidate image against the product
name before it is accepted, and the accepted gallery is saved to `images/<product>/` together with
a `metadata.json`.

## Prerequisites

- Windows 10 or 11 (other platforms should work but are not documented here)
- **Python 3.12 or newer**
- A [SerpApi](https://serpapi.com) API key
- An [NVIDIA Build](https://build.nvidia.com) API key (used for the Nemotron models)

No browser needs to be installed for the app: nothing in the current pipeline launches one.
(`playwright` is still listed as a dependency because several modules import its types.)

## Setup (Windows)

Open PowerShell in the project folder.

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the app and its dependencies
python -m pip install --upgrade pip
python -m pip install -e .

# 3. Install the development tools (pytest, Ruff, mypy)
python -m pip install --group dev

# 4. Create your environment file
Copy-Item .env.example .env
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd` and
`.venv\Scripts\activate.bat` instead.

With [uv](https://docs.astral.sh/uv/) you can instead run `uv sync`.

## Environment variables

Edit `.env` and replace the placeholders with your own keys. `.env` is ignored by Git; only
`.env.example` (placeholders, no real values) is committed.

| Variable | Required | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | Yes | NVIDIA Build key for the Nemotron models (classification and image verification) |
| `PIR_SERPAPI_API_KEY` | Yes | SerpApi key for Google Shopping, retailer-link resolution and Google Images |
| `PIR_REQUEST_TIMEOUT_SECONDS` | No | Per-request timeout for SerpApi calls. Default 15; `.env.example` suggests 30 |

All other settings (`config.py`) have working defaults.

## Run the app

```powershell
streamlit run app.py
```

Streamlit opens `http://localhost:8501`. In the **Shop & Compare** tab, type a product name (for
example `Tata Salt Iodized Salt 1kg`) and press **Compare offers**. Each search makes live
SerpApi, NVIDIA and retailer requests and uses SerpApi credits.

A command-line version of the Single product pipeline is also available:

```powershell
python run.py
```

## Run the tests and linter

```powershell
python -m pytest
python -m ruff check .
```

The test suite makes no network requests: `tests/conftest.py` installs a guard that fails any test
which tries to reach a non-local host, and all providers are faked. Type checking is available with
`python -m mypy .`; it currently reports findings and is non-blocking in CI (see
[docs/mypy-remediation.md](docs/mypy-remediation.md)).

## Project layout

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI (both tabs) |
| `run.py` | CLI entry point and the composition root that wires providers together |
| `shopping_comparison_stage.py` | Shop & Compare orchestration and ranking |
| `shopping_retailers.py` | Approved-retailer allowlist and name matching |
| `product_page_text.py` | Bounded text extraction from retailer product pages |
| `search/retailers/google_shopping_serpapi.py` | Google Shopping search and merchant-link resolution |
| `classifier/nemotron.py`, `prompts.py` | Nemotron client and prompts |
| `models.py`, `config.py`, `interfaces.py` | Shared models, settings and abstract ports |
| `retailer_gallery_stage.py`, `selection_stage.py`, `verification_overrides.py` | Single product image pipeline |
| `search/`, `download/`, `storage/`, `utils/` | Providers and helpers |
| `batch/` | Legacy Excel batch code; not used by the app (see `HANDOFF.md`) |
| `tests/` | Offline test suite |

## Limitations

- **Google Shopping results vary.** The same query can return different sellers, prices and
  ordering from one run to the next, and some listings have no price or no direct retailer link.
  Only the first 10 approved candidates are examined, so a good offer further down Google's list
  can be missed.
- **NVIDIA API rate limits.** Nemotron requests are occasionally throttled or return transient
  errors. The app retries, but a batch that still fails is reported as "couldn't be verified"
  rather than guessed.
- **LLM judgement is not perfect.** Brand-only or very generic queries can be classified
  inconsistently, and size or pack-count differences are sometimes borderline.
- **Retailer pages are optional evidence.** Some retailers block automated page fetches or render
  content client-side; classification then falls back to the listing title and thumbnail.
- **Prices are as listed by Google Shopping** at search time and are not re-verified on the
  retailer site.
- **Region.** Searches target India (`gl=in`, `google.co.in`) and the allowlist is Indian retailers.
- **Credits and latency.** Each comparison uses SerpApi credits and can take a while because candidates are fetched and classified in batches.

## Documentation

[HANDOFF.md](HANDOFF.md) describes the current architecture, design decisions and known issues in
more detail. Older, superseded notes are kept in [docs/archive/](docs/archive/).

## License

Released under the [MIT License](LICENSE).
