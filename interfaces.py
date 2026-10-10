"""
Abstract interfaces (ports) for Shop & Compare.

This is the classic "ports and adapters" (hexagonal) pattern:
    - Ports    = the ABCs below.
    - Adapters = the concrete provider classes that subclass them.

Every port here earns its place: each one has a real implementation AND is
used as a type annotation by at least one caller, so the abstraction is
actually doing dependency-inversion work rather than existing on spec.

| Port                  | Implementation                          | Annotated in                                    |
|-----------------------|-----------------------------------------|-------------------------------------------------|
| `QueryGenerator`      | `search.query_generator.SimpleQueryGenerator` | `selection_stage.py`                       |
| `ImageSearchProvider` | `search.serpapi.SerpApiImageSearchProvider`   | `selection_stage.py`                       |
| `ImageDownloader`     | `download.downloader.HttpImageDownloader`     | `selection_stage.py`, `retailer_gallery_stage.py` |
| `StorageProvider`     | `storage.local_filesystem.LocalFilesystemStorage` | `selection_stage.py`                   |

A `VisionModel` port previously lived here too and was removed: the sole
vision adapter (`classifier.nemotron.NemotronVisionModel`) provides two
capabilities - comparative multi-image ranking and single-image product-match
verification - that are specific enough that neither fits a generic
"classify one image" contract, and both orchestrators annotate that class
concretely. Nothing typed against the port, and its one abstract method had
no callers, so it was pure ceremony. Reintroduce it only if a second vision
provider actually appears AND a caller genuinely needs to be provider-agnostic.

No implementation logic belongs in this file. Every method body here is
either `...` (abstract) or a docstring — nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models import (
    DownloadedImage,
    ImageCandidate,
    ProductImageResult,
    SearchQuery,
)


class QueryGenerator(ABC):
    """
    Turns a raw product name into a diverse set of search queries.

    Implementations may be rule-based (string templates: "{name} front image",
    "{name} back label") or LLM-based (asking a language model to brainstorm
    disambiguating queries). Either way, the pipeline only ever sees
    `list[SearchQuery]`.
    """

    @abstractmethod
    async def generate(self, product_name: str, max_queries: int) -> list[SearchQuery]:
        """Return up to `max_queries` distinct SearchQuery objects for `product_name`."""
        ...


class ImageSearchProvider(ABC):
    """
    A source of candidate image URLs for a given query.

    One adapter per source: Google Images, Bing Images, SerpAPI, Brave
    Search, Amazon, Flipkart, a manufacturer's own site, etc. The pipeline
    treats all of them identically and may fan a query out to several
    providers concurrently.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry key for this provider, e.g. 'google_images'."""
        ...

    @abstractmethod
    async def search(self, query: SearchQuery, max_results: int) -> list[ImageCandidate]:
        """Return up to `max_results` ImageCandidate objects for `query`."""
        ...


class ImageDownloader(ABC):
    """
    Fetches a candidate image's bytes and persists them to local disk.

    Kept separate from `ImageSearchProvider` because download concerns
    (retries, timeouts, content-type sniffing, size limits) are orthogonal to
    *where* a URL came from — one implementation can serve every search
    provider.
    """

    @abstractmethod
    async def download(self, candidate: ImageCandidate, destination_dir: Path) -> DownloadedImage:
        """Fetch `candidate.url` and save it under `destination_dir`."""
        ...


class StorageProvider(ABC):
    """
    Persists final chosen images and/or the pipeline's result record.

    Abstracts *where* results end up: local filesystem, S3/GCS/Azure Blob, a
    database row, etc. Callers never touch a filesystem path or bucket name
    directly — they only call this interface.
    """

    @abstractmethod
    async def persist_image(self, downloaded: DownloadedImage, product_name: str, side: str) -> Path:
        """Move/copy a downloaded image into permanent storage; return its final path."""
        ...

    @abstractmethod
    async def persist_result(self, result: ProductImageResult) -> None:
        """Durably record the final ProductImageResult (e.g. as JSON, or a DB row)."""
        ...
