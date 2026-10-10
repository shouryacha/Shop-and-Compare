"""
Focused tests for the exception-handling fix in `selection_stage.py`'s
`_search_download_select()`.

Before the fix, `self._search_provider.search(...)` was the one external
call in this file with no try/except around it (every other external call -
`_download_one`, `select_best_candidate` - already skip-and-continue on
failure). A real, live SerpAPI read-timeout during end-to-end testing hit
exactly this gap and aborted the whole `ProductImageSelectionStage.run()`
(both front AND back) instead of degrading gracefully for just the affected
side. These tests cover: the search failure is now contained to one side,
`run()` still returns a usable partial (or empty) result instead of raising,
and the existing successful path is unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

from interfaces import ImageDownloader, ImageSearchProvider, QueryGenerator, StorageProvider
from models import (
    ClassificationVerdict,
    DownloadedImage,
    ImageCandidate,
    ImageSide,
    ProductImageResult,
    SearchIntent,
    SearchQuery,
)
from selection_stage import ProductImageSelectionStage

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


class _FakeQueryGenerator(QueryGenerator):
    async def generate(self, product_name: str, max_queries: int) -> list[SearchQuery]:
        return [
            SearchQuery(text=f"{product_name} front", intent=SearchIntent.FRONT),
            SearchQuery(text=f"{product_name} back", intent=SearchIntent.BACK),
        ]


class _RaisingSearchProvider(ImageSearchProvider):
    """Raises on every `search()` call - simulates a network/API failure."""

    @property
    def name(self) -> str:
        return "raising_provider"

    async def search(self, query: SearchQuery, max_results: int) -> list[ImageCandidate]:
        raise RuntimeError("simulated network error contacting the search provider")


class _RaisingForFrontOnlySearchProvider(ImageSearchProvider):
    """Raises only for FRONT-intent queries; succeeds normally for BACK."""

    @property
    def name(self) -> str:
        return "partial_provider"

    async def search(self, query: SearchQuery, max_results: int) -> list[ImageCandidate]:
        if query.intent == SearchIntent.FRONT:
            raise RuntimeError("simulated timeout on the front-side search")
        return [
            ImageCandidate(
                url="https://example.com/back.jpg", source_provider=self.name, source_query=query
            )
        ]


class _FakeSearchProvider(ImageSearchProvider):
    """Always returns exactly one candidate - the existing successful path."""

    @property
    def name(self) -> str:
        return "fake_provider"

    async def search(self, query: SearchQuery, max_results: int) -> list[ImageCandidate]:
        return [
            ImageCandidate(
                url="https://example.com/candidate.jpg", source_provider=self.name, source_query=query
            )
        ]


class _FakeDownloader(ImageDownloader):
    async def download(self, candidate: ImageCandidate, destination_dir: Path) -> DownloadedImage:
        return DownloadedImage(candidate=candidate, local_path=Path("/fake/downloaded.jpg"))


class _FakeStorage(StorageProvider):
    async def persist_image(self, downloaded: DownloadedImage, product_name: str, side: str) -> Path:
        return Path(f"/fake/images/{product_name}/{side}.jpg")

    async def persist_result(self, result: ProductImageResult) -> None:
        return None


class _FakeVisionModel:
    """Stands in for `NemotronVisionModel` - only `select_best_candidate` is used here."""

    name = "fake_vision_model"

    async def select_best_candidate(
        self, image_paths: list[Path], product_name: str, side: ImageSide
    ) -> tuple[int, ClassificationVerdict]:
        verdict = ClassificationVerdict(
            is_correct_product=True, side=side, confidence=0.9, reasoning="fake", model_name=self.name
        )
        return 0, verdict


def _make_stage(search_provider: ImageSearchProvider) -> ProductImageSelectionStage:
    return ProductImageSelectionStage(
        search_provider=search_provider,
        downloader=_FakeDownloader(),
        vision_model=_FakeVisionModel(),  # type: ignore[arg-type]
        storage=_FakeStorage(),
        query_generator=_FakeQueryGenerator(),
    )


# --- _search_download_select (unit level) -----------------------------------


def test_search_download_select_returns_none_when_search_raises(tmp_path: Path) -> None:
    stage = _make_stage(_RaisingSearchProvider())
    query = SearchQuery(text="some product front", intent=SearchIntent.FRONT)

    result = _run(stage._search_download_select(query, "some product", "front", tmp_path))

    assert result is None


def test_search_download_select_still_works_when_search_succeeds(tmp_path: Path) -> None:
    stage = _make_stage(_FakeSearchProvider())
    query = SearchQuery(text="some product front", intent=SearchIntent.FRONT)

    result = _run(stage._search_download_select(query, "some product", "front", tmp_path))

    assert result is not None
    assert result.verdict.confidence == 0.9
    assert result.verdict.is_correct_product is True


# --- run() (integration level: does the failure stay contained to one side?) -


def test_run_degrades_gracefully_when_front_search_fails_but_back_succeeds() -> None:
    stage = _make_stage(_RaisingForFrontOnlySearchProvider())

    result = _run(stage.run("some product"))

    assert result.front_image_path is None
    assert result.front_confidence is None
    assert result.back_image_path is not None
    assert result.back_confidence == 0.9
    assert "front_source" not in result.provider_used
    assert result.provider_used["back_source"] == "partial_provider"


def test_run_returns_empty_result_without_raising_when_both_sides_fail() -> None:
    stage = _make_stage(_RaisingSearchProvider())

    result = _run(stage.run("some product"))

    assert result.front_image_path is None
    assert result.back_image_path is None


def test_run_still_succeeds_normally_on_both_sides() -> None:
    stage = _make_stage(_FakeSearchProvider())

    result = _run(stage.run("some product"))

    assert result.front_image_path is not None
    assert result.back_image_path is not None
    assert result.front_confidence == 0.9
    assert result.back_confidence == 0.9
    assert result.provider_used["front_source"] == "fake_provider"
    assert result.provider_used["back_source"] == "fake_provider"
