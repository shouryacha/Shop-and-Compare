"""
Sequential batch orchestrator.

Runs the EXISTING retailer-first pipeline (`RetailerGalleryStage`, built via
`run.build_stage()` and passed in unchanged) once per `ProductRow`, using
ONLY the product description as the search term (never the EAN - the EAN is
used solely for naming the output ZIP and identifying the row in
results.xlsx). Each row's already-persisted output folder
(`storage.product_dir(product_description)`) is then packaged into a ZIP.

`row.brand` (if the source Excel had a Brand column - see
`column_detection.detect_brand_column`) is passed through to `stage.run()`
as `erp_brand`, purely for `verification_overrides.py`'s deterministic
post-verification brand check - it is never used for search or sent to the
vision model.

Rows are processed one at a time, in order - no concurrency, matching the
existing single-product pipeline's own sequential retailer loop. A failure
on one row (a validation skip, or an unexpected exception from the pipeline)
never stops the batch; it's recorded and the next row is attempted.

No retrieval/verification/storage logic lives here - this module only
sequences existing calls (`stage.run()`, `describe_result()`,
`storage.product_dir()`) and packages their output.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path

from retailer_gallery_stage import RetailerGalleryStage
from run import describe_result
from storage.local_filesystem import LocalFilesystemStorage

from .models import BatchRowResult, ProductRow
from .packaging import zip_product_directory

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


def _count_images(product_dir: Path) -> int:
    if not product_dir.exists():
        return 0
    return sum(1 for f in product_dir.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES)


def _source_label(result) -> str:
    """Reuses `describe_result`'s mode/retailer fields - no new interpretation of `ProductImageResult`."""
    info = describe_result(result)
    if info["mode"] == "retailer_gallery":
        return info["retailer"] or ""
    if info["mode"] == "no_result_fallback_disabled":
        return "None (fallback disabled)"
    return "Google Images (fallback)"


async def run_batch(
    rows: list[ProductRow],
    stage: RetailerGalleryStage,
    storage: LocalFilesystemStorage,
    zips_dir: Path,
) -> AsyncIterator[BatchRowResult]:
    """
    Process `rows` sequentially, yielding one `BatchRowResult` per row as
    soon as that row finishes (skipped, errored, or completed) - so callers
    can render progress incrementally instead of waiting for the whole batch.

    Args:
        rows: Parsed from the uploaded Excel file (see `excel_import.py`).
        stage: The existing `RetailerGalleryStage`, constructed once by the
            caller via `run.build_stage()` and reused for every row.
        storage: The same `LocalFilesystemStorage` instance `stage` persists
            through, used here only to locate each row's output folder.
        zips_dir: Where per-product ZIPs are written.
    """
    for row in rows:
        if not row.product_description:
            yield BatchRowResult(
                row_number=row.row_number,
                ean=row.ean,
                product_description=row.product_description,
                status="Skipped",
                source_used="",
                image_count=0,
                notes="Missing product description",
                zip_path=None,
            )
            continue

        if not row.ean:
            yield BatchRowResult(
                row_number=row.row_number,
                ean=row.ean,
                product_description=row.product_description,
                status="Skipped",
                source_used="",
                image_count=0,
                notes="Missing EAN",
                zip_path=None,
            )
            continue

        try:
            result = await stage.run(row.product_description, erp_brand=row.brand or None)
        except Exception as exc:  # noqa: BLE001 - continue processing remaining rows
            logger.warning("Row %d (EAN=%s) failed: %s", row.row_number, row.ean, exc)
            yield BatchRowResult(
                row_number=row.row_number,
                ean=row.ean,
                product_description=row.product_description,
                status="Error",
                source_used="",
                image_count=0,
                notes=str(exc),
                zip_path=None,
            )
            continue

        product_dir = storage.product_dir(row.product_description)
        image_count = _count_images(product_dir)

        # `product_dir` is only ever created by `storage.persist_gallery()`
        # (see LocalFilesystemStorage.product_dir's own docstring) - never
        # by `stage.run()` on a genuine miss (e.g. all 3 retailers failed,
        # or the Google fallback is disabled - see
        # retailer_gallery_stage.ENABLE_GOOGLE_FALLBACK). Calling
        # `zip_product_directory` on a directory that was never created
        # raises FileNotFoundError; skipping it here (rather than letting
        # that exception escape this row) is what makes "no images found
        # for this product" move on to the next row instead of aborting the
        # whole batch.
        zip_path = None
        if image_count > 0:
            try:
                zip_path = zip_product_directory(product_dir, row.ean, zips_dir)
            except Exception as exc:  # noqa: BLE001 - continue processing remaining rows
                logger.warning("Row %d (EAN=%s): failed to zip %s: %s", row.row_number, row.ean, product_dir, exc)
                yield BatchRowResult(
                    row_number=row.row_number,
                    ean=row.ean,
                    product_description=row.product_description,
                    status="Error",
                    source_used=_source_label(result),
                    image_count=image_count,
                    notes=f"Found {image_count} image(s) but failed to zip them: {exc}",
                    zip_path=None,
                )
                continue

        yield BatchRowResult(
            row_number=row.row_number,
            ean=row.ean,
            product_description=row.product_description,
            status="Success" if image_count > 0 else "No Images Found",
            source_used=_source_label(result),
            image_count=image_count,
            notes="",
            zip_path=zip_path,
        )
