"""
Local filesystem storage.

Implements `interfaces.StorageProvider`: persists the winning front/back
images as `front.<ext>` / `back.<ext>` inside a per-product folder under
`config.Settings.images_dir` (the single image root - there is no separate
output directory), and writes the final `ProductImageResult` as
`metadata.json` in that same folder.

`persist_gallery` is an additional, non-interface method for the retailer-
gallery workflow (`retailer_gallery_stage.py`): saves a full, ordered list
of images as `image_01.<ext>, image_02.<ext>, ...` in that same per-product
folder, with no front/back/side judgement at all - the retailer's gallery
order is preserved exactly.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from config import get_settings
from interfaces import StorageProvider
from models import DownloadedImage, ProductImageResult
from utils.slugify import slugify

_GALLERY_FILENAME_WIDTH = 2  # image_01.jpg, not image_1.jpg

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"


class LocalFilesystemStorage(StorageProvider):
    """`StorageProvider` adapter that writes to the local filesystem."""

    def __init__(self, images_dir: Path | None = None) -> None:
        """Args: images_dir: Root image directory; defaults to `config.Settings.images_dir`."""
        self._images_dir = images_dir or get_settings().images_dir

    async def persist_image(self, downloaded: DownloadedImage, product_name: str, side: str) -> Path:
        """Copy `downloaded.local_path` to `<images_dir>/<product_slug>/<side><ext>`."""
        product_dir = self.product_dir(product_name)
        product_dir.mkdir(parents=True, exist_ok=True)

        extension = downloaded.local_path.suffix or ".jpg"
        final_path = product_dir / f"{side}{extension}"
        shutil.copy2(downloaded.local_path, final_path)

        logger.info("Persisted %s image for %r -> %s", side, product_name, final_path)
        return final_path

    async def persist_gallery(
        self, downloaded_images: list[DownloadedImage], product_name: str
    ) -> list[Path]:
        """
        Copy every image in `downloaded_images`, in order, to
        `<images_dir>/<product_slug>/image_01.<ext>`,
        `image_02.<ext>`, ... - preserving the retailer's gallery order
        exactly, with no front/back/side judgement.

        Returns the persisted paths, in the same order as `downloaded_images`.
        """
        product_dir = self.product_dir(product_name)
        product_dir.mkdir(parents=True, exist_ok=True)

        persisted_paths: list[Path] = []
        for index, downloaded in enumerate(downloaded_images, start=1):
            extension = downloaded.local_path.suffix or ".jpg"
            final_path = product_dir / f"image_{index:0{_GALLERY_FILENAME_WIDTH}d}{extension}"
            shutil.copy2(downloaded.local_path, final_path)
            persisted_paths.append(final_path)

        logger.info(
            "Persisted %d gallery image(s) for %r -> %s", len(persisted_paths), product_name, product_dir
        )
        return persisted_paths

    async def persist_result(self, result: ProductImageResult) -> None:
        """Write `result` as `metadata.json` inside the product's output folder."""
        product_dir = self.product_dir(result.product_name)
        product_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = product_dir / _METADATA_FILENAME
        metadata_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Wrote metadata for %r -> %s", result.product_name, metadata_path)

    def product_dir(self, product_name: str) -> Path:
        """Return the per-product image folder for `product_name`."""
        return self._images_dir / slugify(product_name)
