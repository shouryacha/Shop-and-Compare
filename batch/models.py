"""Plain data records shared across the `batch` package - no behavior, just shapes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProductRow:
    """One row read from the uploaded Excel file, before any validation."""

    row_number: int  # 1-based, matching the row's position in the spreadsheet (row 1 = header)
    ean: str  # "" if the cell was blank
    product_description: str  # "" if the cell was blank
    brand: str = ""  # "" if the cell was blank OR the source file has no Brand column at all


@dataclass(frozen=True)
class BatchRowResult:
    """Outcome of processing one `ProductRow` - one of these becomes one line in results.xlsx."""

    row_number: int
    ean: str
    product_description: str
    status: str  # "Success" | "No Images Found" | "Skipped" | "Error"
    source_used: str  # retailer name, "Google Images (fallback)", or "" if skipped/errored
    image_count: int
    notes: str
    zip_path: Path | None  # only set when status == "Success" (i.e. image_count > 0); None otherwise
