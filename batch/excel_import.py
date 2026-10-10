"""
Reads an ERP Excel export into `ProductRow` records.

Only the first worksheet is read. Column position is never assumed - see
`column_detection.py`. Values are read as-is (including blanks); this module
does not decide whether a row is valid, it only extracts what's in the
sheet - `batch/processor.py` decides how to handle missing EAN/description.
"""

from __future__ import annotations

from typing import Any, BinaryIO

import openpyxl

from .column_detection import detect_brand_column, detect_columns
from .models import ProductRow


def _cell_to_text(value: Any) -> str:
    """
    Render any cell value as a clean, trimmed string.

    Handles the common Excel gotcha where a numeric-looking code (like an
    EAN) is stored as a float: a bare `str(12345.0)` would produce
    "12345.0" instead of "12345", so whole-number floats are converted to
    plain integer strings first.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def read_product_rows(source: BinaryIO | str) -> list[ProductRow]:
    """
    Parse the first worksheet of `source` (a path or a file-like object,
    e.g. a Streamlit `UploadedFile`) into `ProductRow` records, one per data
    row, in original spreadsheet order. Fully blank rows are skipped
    silently (they're not product entries); rows with SOME data but a
    missing EAN or description are still returned, so the caller can record
    why they were skipped.

    Raises:
        ColumnDetectionError: if the header row doesn't contain a
            recognizable EAN or Product Description column.
    """
    workbook = openpyxl.load_workbook(source, data_only=True, read_only=True)
    try:
        worksheet = workbook.active
        rows_iter = worksheet.iter_rows(values_only=True)
        header_row = list(next(rows_iter, None) or ())
        ean_index, description_index = detect_columns(header_row)
        brand_index = detect_brand_column(header_row)  # None if the file has no Brand column

        rows: list[ProductRow] = []
        for row_number, row in enumerate(rows_iter, start=2):  # row 1 is the header
            ean = _cell_to_text(row[ean_index]) if ean_index < len(row) else ""
            description = _cell_to_text(row[description_index]) if description_index < len(row) else ""
            brand = _cell_to_text(row[brand_index]) if brand_index is not None and brand_index < len(row) else ""
            if ean == "" and description == "" and all(_cell_to_text(cell) == "" for cell in row):
                continue  # fully blank row - not a product entry
            rows.append(ProductRow(row_number=row_number, ean=ean, product_description=description, brand=brand))
        return rows
    finally:
        workbook.close()
