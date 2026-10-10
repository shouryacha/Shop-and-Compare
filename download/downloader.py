"""
HTTP image downloader.

Given image URLs (e.g. from a `search/` provider), fetches each one, verifies
the bytes actually decode as an image, and saves it under a caller-supplied
destination directory with a unique filename. Implements
`interfaces.ImageDownloader`.

Scope: this module performs ONLY downloading — it does not search for URLs
and does not classify/judge any image. It is not wired to Qwen (or any other
`classifier/` adapter) or the pipeline.

Uses the standard library for the HTTP call (`urllib`, run off the event
loop via `asyncio.to_thread`) to avoid adding a new HTTP client dependency.
Image validity is verified with Pillow (`PIL.Image.verify()`), which
actually decodes enough of the file to catch truncated/corrupt images - a
magic-byte check alone would only catch a wrong file type, not corruption.
Pillow is not yet declared in `pyproject.toml`; see the note at the end of
this module's usage instructions.
"""

from __future__ import annotations

import asyncio
import io
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from interfaces import ImageDownloader
from models import DownloadedImage, ImageCandidate
from utils.hashing import sha256_hex

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: Final[float] = 15.0
_MAX_IMAGE_BYTES: Final[int] = 25 * 1024 * 1024  # 25 MB - defensive cap against runaway downloads
_USER_AGENT: Final[str] = "Mozilla/5.0 (compatible; ProductImageRetriever/0.1)"

_FORMAT_TO_EXTENSION: Final[dict[str, str]] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}


class ImageDownloadError(Exception):
    """Base class for a single image failing to download or validate."""


class InvalidImageUrlError(ImageDownloadError):
    """Raised when a URL is not a well-formed http(s) URL."""


class ImageNetworkError(ImageDownloadError):
    """Raised for connection failures, DNS errors, and non-2xx HTTP responses."""


class ImageTimeoutError(ImageDownloadError):
    """Raised when the request does not complete within the configured timeout."""


class CorruptImageError(ImageDownloadError):
    """Raised when the downloaded bytes do not decode as a valid image."""


@dataclass(frozen=True)
class _SavedImage:
    """Internal result of successfully fetching, verifying, and saving one image."""

    local_path: Path
    content_type: str
    size_bytes: int
    checksum_sha256: str


class HttpImageDownloader(ImageDownloader):
    """`ImageDownloader` adapter that fetches images over plain HTTP(S)."""

    async def download_images(
        self, urls: list[str], destination_dir: Path
    ) -> list[Path]:
        """
        Download each URL in `urls`, skipping duplicates and any URL that
        fails (invalid URL, network error, timeout, corrupt image), and
        return the local paths of the ones that succeeded.

        Args:
            urls: Candidate image URLs, e.g. from a `search/` provider.
            destination_dir: Where to save files.

        Returns:
            Local file paths, in the same relative order as `urls`, for
            every URL that downloaded and verified successfully. Shorter
            than `urls` whenever some were skipped.
        """
        if not urls:
            return []

        unique_urls = self._dedupe_preserve_order(urls)
        if len(unique_urls) < len(urls):
            logger.info("Skipped %d duplicate URL(s)", len(urls) - len(unique_urls))

        saved_paths: list[Path] = []
        for url in unique_urls:
            try:
                saved = await self._download_one(url, destination_dir)
            except ImageDownloadError as exc:
                logger.warning("Skipping %r: %s", url, exc)
                continue

            logger.info("Downloaded %r -> %s (%d bytes)", url, saved.local_path, saved.size_bytes)
            saved_paths.append(saved.local_path)

        logger.info("Downloaded %d/%d requested image(s)", len(saved_paths), len(urls))
        return saved_paths

    async def download(self, candidate: ImageCandidate, destination_dir: Path) -> DownloadedImage:
        """
        Satisfies `interfaces.ImageDownloader`.

        Unlike `download_images`, this downloads exactly one required
        candidate and raises `ImageDownloadError` (or a subclass) on failure
        rather than skipping it.
        """
        saved = await self._download_one(str(candidate.url), destination_dir)
        return DownloadedImage(
            candidate=candidate,
            local_path=saved.local_path,
            content_type=saved.content_type,
            file_size_bytes=saved.size_bytes,
            checksum_sha256=saved.checksum_sha256,
        )

    async def _download_one(self, url: str, destination_dir: Path) -> _SavedImage:
        """Validate, fetch, verify, and save a single URL; raises on any failure."""
        self._validate_url(url)
        data = await asyncio.to_thread(self._fetch_bytes, url)
        image_format = self._verify_image(data, url)
        local_path = self._save_to_disk(data, image_format, destination_dir)
        return _SavedImage(
            local_path=local_path,
            content_type=f"image/{image_format.lower()}",
            size_bytes=len(data),
            checksum_sha256=sha256_hex(data),
        )

    def _validate_url(self, url: str) -> None:
        """Reject anything that isn't a well-formed http(s) URL before attempting a fetch."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise InvalidImageUrlError(f"Not a valid http(s) URL: {url!r}")

    def _fetch_bytes(self, url: str) -> bytes:
        """
        Blocking HTTP GET (invoked via `asyncio.to_thread`), capped at
        `_MAX_IMAGE_BYTES`. Isolated into its own method so the sync/async
        boundary lives in exactly one place.
        """
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=_DEFAULT_TIMEOUT_SECONDS) as response:
                data = response.read(_MAX_IMAGE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise ImageNetworkError(f"HTTP {exc.code} fetching {url!r}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ImageTimeoutError(f"Timed out fetching {url!r}") from exc
            raise ImageNetworkError(f"Network error fetching {url!r}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ImageTimeoutError(f"Timed out fetching {url!r}") from exc
        except OSError as exc:
            raise ImageNetworkError(f"Network error fetching {url!r}: {exc}") from exc

        if not data:
            raise CorruptImageError(f"Empty response body for {url!r}")
        if len(data) > _MAX_IMAGE_BYTES:
            raise ImageDownloadError(
                f"Image at {url!r} exceeds max size of {_MAX_IMAGE_BYTES} bytes"
            )
        return data

    def _verify_image(self, data: bytes, url: str) -> str:
        """
        Decode-verify `data` with Pillow and return its detected format
        (e.g. `"JPEG"`). Raises `CorruptImageError` if it isn't a genuine,
        intact image.
        """
        try:
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
        except UnidentifiedImageError as exc:
            raise CorruptImageError(f"Not a recognizable image ({url!r}): {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - any decode failure means corrupt/invalid
            raise CorruptImageError(f"Image failed verification ({url!r}): {exc}") from exc

        # `Image.verify()` invalidates the file pointer/object for further use,
        # so re-open a fresh instance just to read the confirmed format.
        with Image.open(io.BytesIO(data)) as verified:
            image_format = verified.format

        if not image_format:
            raise CorruptImageError(f"Could not determine image format ({url!r})")
        return image_format

    def _save_to_disk(self, data: bytes, image_format: str, destination_dir: Path) -> Path:
        """Write `data` to `destination_dir` under a newly generated unique filename."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        extension = _FORMAT_TO_EXTENSION.get(image_format, f".{image_format.lower()}")
        local_path = destination_dir / f"{uuid4().hex}{extension}"
        local_path.write_bytes(data)
        return local_path

    @staticmethod
    def _dedupe_preserve_order(urls: list[str]) -> list[str]:
        """Return `urls` with duplicates removed, keeping the first occurrence of each."""
        seen: set[str] = set()
        deduped: list[str] = []
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return deduped
