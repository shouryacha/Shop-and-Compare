"""
CLI entry point for Shop & Compare.

Thin wiring only: prompts for a product name, constructs the concrete
providers (SerpAPI search, HTTP downloader, Nemotron vision model, local
filesystem storage), hands them to `RetailerGalleryStage` - the primary,
retailer-first retrieval workflow, with the existing
`ProductImageSelectionStage` (Google Images comparative-selection workflow)
wired in as its fallback - and prints the result. All actual search/
download/verification/persist logic lives in those existing modules; this
file does not reimplement any of it.

Progress messages for each stage come from the `logging` calls already
present in `retailer_gallery_stage.py`, `selection_stage.py`, and the
providers they use - `configure_logging("INFO")` is what makes them visible
on the console; no separate progress-printing logic is added here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from classifier.nemotron import NemotronVisionModel, VisionModelError
from download.downloader import HttpImageDownloader
from models import ProductImageResult
from retailer_gallery_stage import RetailerGalleryStage
from search.serpapi import SerpApiImageSearchProvider, SerpApiSearchError
from selection_stage import ProductImageSelectionStage
from shopping_comparison_stage import ShoppingComparisonStage
from storage.local_filesystem import LocalFilesystemStorage
from utils.logging_setup import configure_logging


def prompt_for_product_name() -> str:
    """Prompt until a non-empty product name is entered."""
    while True:
        product_name = input("Enter product name: ").strip()
        if product_name:
            return product_name
        print("Product name cannot be empty. Please try again.")


def prompt_for_erp_brand() -> str | None:
    """
    Optional ERP brand, used only for `verification_overrides.py`'s
    deterministic brand-mismatch check (see `RetailerGalleryStage.run`'s
    `erp_brand` parameter) - never sent to search or the vision model.
    Blank input (just pressing Enter) means "don't run that check" for
    this product, exactly as if this prompt didn't exist.
    """
    erp_brand = input("Enter ERP brand (optional, press Enter to skip): ").strip()
    return erp_brand or None


def build_stage() -> tuple[RetailerGalleryStage, LocalFilesystemStorage]:
    """
    Construct the concrete providers required by this task and wire them
    into a `RetailerGalleryStage`, with a `ProductImageSelectionStage` (the
    Google Images workflow) as its fallback. Returns the storage instance
    alongside the stage so its `product_dir()` helper can be reused to
    report the metadata.json location afterwards, instead of duplicating
    that path-construction logic here.
    """
    downloader = HttpImageDownloader()
    vision_model = NemotronVisionModel()
    storage = LocalFilesystemStorage()

    fallback_stage = ProductImageSelectionStage(
        search_provider=SerpApiImageSearchProvider(),
        downloader=downloader,
        vision_model=vision_model,
        storage=storage,
    )

    stage = RetailerGalleryStage(
        downloader=downloader,
        vision_model=vision_model,
        storage=storage,
        fallback_stage=fallback_stage,
    )
    return stage, storage


def build_shopping_stage() -> tuple[ShoppingComparisonStage, LocalFilesystemStorage]:
    """
    Construct a `ShoppingComparisonStage` (the additive "Shop & Compare"
    feature - see `shopping_comparison_stage.py`) with fresh provider
    instances, mirroring `build_stage()` above. Kept as a separate function
    rather than folded into `build_stage()` since the two stages serve
    different use cases (single verified gallery vs. a shopping comparison
    list) and a caller only ever needs one or the other.
    """
    downloader = HttpImageDownloader()
    vision_model = NemotronVisionModel()
    storage = LocalFilesystemStorage()

    stage = ShoppingComparisonStage(
        downloader=downloader, vision_model=vision_model, storage=storage
    )
    return stage, storage


def describe_result(result: ProductImageResult) -> dict[str, Any]:
    """
    Interpret a finished `ProductImageResult`'s shape once, for reuse by
    every frontend (`print_result` below, and `app.py`'s Streamlit UI).

    This is pure display-field-picking - it reads already-computed fields
    off a finished result; it contains no retrieval/verification/search/
    storage behavior of its own.

    Returns a dict with:
        mode: "retailer_gallery", "fallback_front_back", or (only while
            `retailer_gallery_stage.ENABLE_GOOGLE_FALLBACK` is temporarily
            False) "no_result_fallback_disabled".
        retailer / retailer_url: only set when mode == "retailer_gallery".
        image_paths: ordered list of image Paths actually produced - the
            retailer's gallery order for "retailer_gallery", [front, back]
            (skipping whichever side is missing) for "fallback_front_back",
            or empty for "no_result_fallback_disabled".
    """
    retailer = result.metadata.get("retailer")
    if retailer:
        return {
            "mode": "retailer_gallery",
            "retailer": retailer,
            "retailer_url": result.metadata.get("retailer_product_url"),
            "image_paths": [Path(p) for p in result.metadata.get("gallery_image_paths", [])],
        }
    if result.metadata.get("fallback_disabled"):
        return {
            "mode": "no_result_fallback_disabled",
            "retailer": None,
            "retailer_url": None,
            "image_paths": [],
        }
    return {
        "mode": "fallback_front_back",
        "retailer": None,
        "retailer_url": None,
        "image_paths": [p for p in (result.front_image_path, result.back_image_path) if p],
    }


def print_result(result: ProductImageResult, metadata_path: Path) -> None:
    """Print a summary appropriate to which workflow actually produced `result`."""
    info = describe_result(result)

    print("\n" + "=" * 60)
    if info["mode"] == "no_result_fallback_disabled":
        print("Retrieval failed (no result)")
    else:
        print("Retrieval complete")
    print("=" * 60)
    print(f"Product name : {result.product_name}")

    if info["mode"] == "retailer_gallery":
        print(f"Source       : retailer gallery ({info['retailer']})")
        print(f"Retailer URL : {info['retailer_url']}")
        print(f"Gallery size : {len(info['image_paths'])} image(s)")
        for i, path in enumerate(info["image_paths"], start=1):
            print(f"  {i}. {path}")
    elif info["mode"] == "no_result_fallback_disabled":
        print("Source       : none - Amazon and Google Shopping both failed")
        print("               (Google/SerpAPI fallback is temporarily disabled -")
        print("               see ENABLE_GOOGLE_FALLBACK in retailer_gallery_stage.py)")
    else:
        print("Source       : Google Images fallback (front/back)")
        print(f"Front image  : {result.front_image_path or 'not found'}")
        print(f"Back image   : {result.back_image_path or 'not found'}")

    print(f"Metadata     : {metadata_path}")


def main() -> None:
    configure_logging("INFO")

    product_name = prompt_for_product_name()
    erp_brand = prompt_for_erp_brand()
    print(f"\nStarting retrieval for: {product_name}\n")

    try:
        stage, storage = build_stage()
    except (SerpApiSearchError, VisionModelError, ValueError) as exc:
        print(f"Could not initialize a required provider: {exc}")
        sys.exit(1)

    result = asyncio.run(stage.run(product_name, erp_brand=erp_brand))

    metadata_path = storage.product_dir(product_name) / "metadata.json"
    print_result(result, metadata_path)


if __name__ == "__main__":
    main()
