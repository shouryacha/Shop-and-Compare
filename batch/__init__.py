"""
Excel batch-processing package for Shop & Compare.

Runs the EXISTING retrieval pipeline (`RetailerGalleryStage`, constructed via
`run.build_stage()`, unchanged) once per product row read from an ERP Excel
export, then packages each product's already-persisted output into a ZIP.

No retrieval, verification, search, download, or storage logic lives here -
this package only reads Excel input, sequences existing pipeline calls, and
packages/report their output.

Modules:
    column_detection - finds the EAN / Product Description columns in an
        arbitrary ERP export header row, tolerant of header naming variations.
    excel_import     - reads an .xlsx file into `ProductRow` records.
    processor         - sequential orchestrator: runs the existing pipeline
        once per row and yields a `BatchRowResult` as each row finishes.
    packaging         - per-product ZIPs, the results.xlsx summary, and the
        final bundle ZIP.
    models            - `ProductRow` / `BatchRowResult` records shared by the
        modules above.
"""
