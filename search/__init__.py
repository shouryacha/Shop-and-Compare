"""
Search: retailer-native search (the primary path) and the Google Images
fallback (via SerpAPI).

Contents:
    - `retailers/` - the Amazon and Google Shopping adapters (both
      SerpApi-backed), each exposing `search_product_page(page, query) -> list[str]`
      (candidate product URLs, own ranking order) and
      `extract_gallery(page, url)`. Not driven through `ImageSearchProvider`
      - see `retailers/__init__.py`.
    - `serpapi.py` - `SerpApiImageSearchProvider`, implementing
      `interfaces.ImageSearchProvider`. This is the concrete search
      provider used by `selection_stage.ProductImageSelectionStage`, the
      Google Images fallback that `retailer_gallery_stage.RetailerGalleryStage`
      only calls after Amazon and Google Shopping have both failed.
    - `query_generator.py` - `SimpleQueryGenerator`, used by
      `selection_stage.py` to build the "<product name> <side> image"
      queries sent to the fallback search provider.
    - `query_normalizer.py` - `normalize_for_search()`, used by
      `retailer_gallery_stage.py` to strip ERP packaging/inventory codes
      (PP, PCH, PET, BTL, ...) before querying each retailer's own search box.
      Search-only; verification always uses the original, unmodified ERP name.

Wiring for all of the above lives in `run.build_stage()`, the single
composition root reused by the CLI and the Streamlit app.
"""
