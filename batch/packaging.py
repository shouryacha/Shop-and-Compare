"""
Packaging for the batch workflow: per-product ZIPs, the results.xlsx
summary, and the final bundle ZIP.

Pure packaging/reporting of output the existing pipeline already persisted
to disk (via `storage.local_filesystem.LocalFilesystemStorage`, unchanged) -
no retrieval, verification, or storage logic lives here.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import openpyxl

from .models import BatchRowResult

_RESULTS_COLUMNS = ["EAN", "Product Description", "Status", "Source Used", "Number of Images", "Notes"]

# Defensive only: EANs are expected to already be filesystem-safe (digits),
# but this guards against stray characters from a messy ERP export ending
# up in a filename.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/*?:"<>|]')


def _safe_zip_name(ean: str) -> str:
    return _UNSAFE_FILENAME_CHARS.sub("_", ean).strip() or "unknown_ean"


def zip_product_directory(product_dir: Path, ean: str, dest_dir: Path) -> Path:
    """
    Zip every file directly inside `product_dir` (no nested folder, per
    spec: `image_01.jpg` etc. sit at the ZIP root) as `<dest_dir>/<ean>.zip`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{_safe_zip_name(ean)}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(product_dir.iterdir()):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.name)
    return zip_path


def write_results_workbook(results: list[BatchRowResult], dest_path: Path) -> None:
    """Write the batch run summary as results.xlsx."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    sheet.append(_RESULTS_COLUMNS)
    for result in results:
        sheet.append(
            [
                result.ean,
                result.product_description,
                result.status,
                result.source_used,
                result.image_count,
                result.notes,
            ]
        )
    workbook.save(dest_path)


def build_final_bundle(results_xlsx_path: Path, product_zip_paths: list[Path], dest_path: Path) -> Path:
    """Bundle results.xlsx plus every per-product ZIP into one final ZIP."""
    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(results_xlsx_path, arcname=results_xlsx_path.name)
        for zip_path in product_zip_paths:
            archive.write(zip_path, arcname=zip_path.name)
    return dest_path
