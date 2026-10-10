"""
Image selection stage: the end-to-end front/back retrieval workflow for one
product.

Concrete, dependency-injected workflow, run twice - once for "front", once
for "back" - as two fully independent passes:
    1. Search for "<product name> <side> image".
    2. Download up to `candidates_per_side` candidates into their own
       per-side folder inside a private, per-run OS temp directory
       (`<tmp_root>/front/`, `<tmp_root>/back/`) - never under `images/`.
    3. Send ALL of those candidates to the vision model together in a
       single comparative request - never classified independently - and
       let it rank them and pick the single best match.
The front and back candidate pools are never merged: a back-search result
can never win the front slot, or vice versa - each side gets its own
`select_best_candidate` call over only its own downloaded candidates. The
winners are persisted via the injected StorageProvider as `front.<ext>` /
`back.<ext>` plus a `metadata.json` under `<images_dir>/<product_slug>/`.
The whole temp directory (winners and losers alike) is removed once both
sides have been persisted.

This stage depends on `classifier.nemotron.NemotronVisionModel` concretely,
because comparative multi-image ranking in a single request is specific to
that adapter's `select_best_candidate`. Search, download, query generation,
and storage remain fully interface-driven and swappable (see
`interfaces.py`).

Reached only as `retailer_gallery_stage.RetailerGalleryStage`'s fallback,
after Amazon and Google Shopping have both failed to produce an accepted
match.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from classifier.nemotron import NemotronVisionModel
from interfaces import ImageDownloader, ImageSearchProvider, QueryGenerator, StorageProvider
from models import (
    ClassifiedImage,
    DownloadedImage,
    ImageCandidate,
    ImageSide,
    ProductImageResult,
    SearchQuery,
)
from search.query_generator import SimpleQueryGenerator

logger = logging.getLogger(__name__)


class ProductImageSelectionStage:
    """Runs the front/back search -> download -> comparative-select -> persist workflow."""

    def __init__(
        self,
        search_provider: ImageSearchProvider,
        downloader: ImageDownloader,
        vision_model: NemotronVisionModel,
        storage: StorageProvider,
        query_generator: QueryGenerator | None = None,
        *,
        candidates_per_side: int = 5,
        min_confidence: float = 0.0,
    ) -> None:
        """
        Args:
            vision_model: Concretely `NemotronVisionModel` (see module
                docstring) - comparative multi-image ranking is specific to
                that adapter, so there is no abstract vision port to type
                against (see `interfaces.py`).
            candidates_per_side: How many candidates to download and send in
                a single comparative request per side ("top N"). Defaults to
                5, per spec, which is also the empirically-verified maximum
                images per request for this model
                (`classifier.nemotron._MAX_IMAGES_PER_REQUEST`).
            min_confidence: If the model's chosen candidate for a side has
                `confidence` below this floor, that side is treated as
                unresolved (no image persisted) rather than accepting a
                low-confidence pick. Defaults to 0.0 (no floor) - the
                comparative prompt has no explicit "none of these qualify"
                option, so it always returns its best guess among what was
                given; this is a safety net on top of that.
        """
        self._search_provider = search_provider
        self._downloader = downloader
        self._vision_model = vision_model
        self._storage = storage
        self._query_generator = query_generator or SimpleQueryGenerator()
        self._candidates_per_side = candidates_per_side
        self._min_confidence = min_confidence

    async def run(self, product_name: str) -> ProductImageResult:
        """
        Execute the front pass and the back pass as two fully independent
        comparative selections, then persist both winners, and return the
        result.
        """
        logger.info("Starting image selection stage for product=%r", product_name)

        queries = await self._query_generator.generate(product_name, max_queries=2)
        front_query, back_query = queries[0], queries[1]

        # Candidates for both sides are downloaded into a private, per-run OS
        # temp directory - never under `images/` - and removed once the
        # winners have been copied into permanent storage below.
        with tempfile.TemporaryDirectory(prefix="pir_selection_") as tmp_dir:
            tmp_root = Path(tmp_dir)

            # Each side is searched, downloaded, and compared entirely on its
            # own - never merged - so a back-search result can never win the
            # front slot, or vice versa.
            best_front = await self._search_download_select(
                front_query, product_name, side="front", destination_dir=tmp_root / "front"
            )
            best_back = await self._search_download_select(
                back_query, product_name, side="back", destination_dir=tmp_root / "back"
            )
            logger.info(
                "Selected best_front=%s best_back=%s",
                best_front.verdict.confidence if best_front else None,
                best_back.verdict.confidence if best_back else None,
            )

            result = await self._persist(product_name, best_front, best_back)

        logger.info(
            "Finished image selection stage for product=%r front=%s back=%s",
            product_name,
            result.front_image_path,
            result.back_image_path,
        )
        return result

    async def _search_download_select(
        self, query: SearchQuery, product_name: str, side: str, destination_dir: Path
    ) -> ClassifiedImage | None:
        """
        Search, download up to `candidates_per_side` candidates into
        `destination_dir`, then send all of them to the vision model
        together in a single comparative request.

        Returns `None` if the search itself failed, nothing downloaded
        successfully, the comparative request itself failed, or the winning
        candidate's confidence fell below `min_confidence`.
        """
        try:
            candidates = await self._search_provider.search(query, self._candidates_per_side)
        except Exception as exc:  # noqa: BLE001 - skip-and-continue, matches select_best_candidate below
            logger.warning("Search failed for side=%s query=%r: %s", side, query.text, exc)
            return None

        logger.info("Found %d candidate(s) for query=%r", len(candidates), query.text)

        downloaded = await self._download_all(candidates, destination_dir)

        if not downloaded:
            logger.warning(
                "No candidates downloaded successfully for side=%s; nothing to compare", side
            )
            return None

        try:
            zero_based_index, verdict = await self._vision_model.select_best_candidate(
                [d.local_path for d in downloaded], product_name, ImageSide(side)
            )
        except Exception as exc:  # noqa: BLE001 - skip-and-continue, matches pipeline.py
            logger.warning("Comparative selection failed for side=%s: %s", side, exc)
            return None

        if verdict.confidence < self._min_confidence:
            logger.info(
                "Best candidate for side=%s had confidence=%.2f below floor=%.2f; rejecting",
                side,
                verdict.confidence,
                self._min_confidence,
            )
            return None

        return ClassifiedImage(downloaded=downloaded[zero_based_index], verdict=verdict)

    async def _download_all(
        self, candidates: list[ImageCandidate], destination_dir: Path
    ) -> list[DownloadedImage]:
        """Download every candidate into `destination_dir`, skipping individual errors."""

        async def download_one(candidate: ImageCandidate) -> DownloadedImage | None:
            try:
                return await self._downloader.download(candidate, destination_dir=destination_dir)
            except Exception as exc:  # noqa: BLE001 - skip-and-continue, matches pipeline.py
                logger.warning("Skipping candidate url=%s: %s", candidate.url, exc)
                return None

        results = await asyncio.gather(*(download_one(c) for c in candidates))
        return [r for r in results if r is not None]

    async def _persist(
        self,
        product_name: str,
        best_front: ClassifiedImage | None,
        best_back: ClassifiedImage | None,
    ) -> ProductImageResult:
        """Save the winning image(s), build metadata.json content, and persist both."""
        front_path = back_path = None
        metadata: dict[str, Any] = {}
        provider_used = {"search": self._search_provider.name, "vision": self._vision_model.name}

        if best_front is not None:
            front_path = await self._storage.persist_image(
                best_front.downloaded, product_name, "front"
            )
            metadata["front"] = self._verdict_metadata(best_front)
            provider_used["front_source"] = best_front.downloaded.candidate.source_provider

        if best_back is not None:
            back_path = await self._storage.persist_image(
                best_back.downloaded, product_name, "back"
            )
            metadata["back"] = self._verdict_metadata(best_back)
            provider_used["back_source"] = best_back.downloaded.candidate.source_provider

        result = ProductImageResult(
            product_name=product_name,
            front_image_path=front_path,
            back_image_path=back_path,
            front_confidence=best_front.verdict.confidence if best_front else None,
            back_confidence=best_back.verdict.confidence if best_back else None,
            provider_used=provider_used,
            metadata=metadata,
        )
        await self._storage.persist_result(result)
        return result

    @staticmethod
    def _verdict_metadata(classified: ClassifiedImage) -> dict[str, Any]:
        """
        Pull confidence plus brand/product_name/weight (if the model found
        them) out of a winning verdict for inclusion in metadata.json.
        `models.ClassificationVerdict` has no dedicated fields for those
        three, so they live on `raw_model_response` (see classifier/nemotron.py).
        """
        raw = classified.verdict.raw_model_response
        extra = raw if isinstance(raw, dict) else {}
        return {
            "confidence": classified.verdict.confidence,
            "brand": extra.get("brand"),
            "product_name": extra.get("product_name"),
            "weight": extra.get("weight"),
            "model_used": classified.verdict.model_name,
        }
