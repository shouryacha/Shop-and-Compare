"""
Storage adapters.

Responsibility: persist the winning front/back images and the final
`models.ProductImageResult` somewhere durable. Implements
`interfaces.StorageProvider`.

Conventions for future adapters added here:
    - One module per backend, e.g. `storage/local_filesystem.py` defining
      `class LocalFilesystemStorage(StorageProvider): ...`, mirrored for S3,
      GCS, or a database-backed result store.
    - `persist_image` should be responsible for final naming/organization
      (e.g. `images/{product_slug}/{side}.jpg`) and must return the path/URI
      callers should treat as canonical.
    - `persist_result` should be responsible for serialization format
      (JSON file, DB row, etc.) and idempotency (re-running for the same
      product should not corrupt prior results).
    - Adapters must not know about search, download, or classification —
      they only ever see already-downloaded images and a finished result
      object.

No concrete storage adapter is implemented yet; this is scaffolding only.
"""
