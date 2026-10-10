"""
Unit tests for the candidate-title verification feature added to
`classifier/nemotron.py` / `prompts.py` after a live false-positive: Amazon's
"Charminar Long Grain, 1kg" was accepted as a match for the requested
"Kohinoor Charminar Sel Bsmti Rice 1KG PP" (i.e. "...Select Basmati Rice
1kg"), because verification had no visibility into what the retailer itself
called the pictured product - only the image and the ERP name.

These tests cover what's mockable/deterministic:
    - the candidate title is correctly threaded from `verify_product_match()`
      into the actual message payload sent for classification (no live model
      call - `NemotronVisionModel._classify` is mocked).
    - the system prompt actually contains the guidance and worked example
      telling the model to treat a differing retailer listing title as a
      meaningful mismatch, not a naming-convention difference.

Whether the live model actually rejects this specific case is an integration
/ live-verification question, not something a mocked unit test can prove -
see the live smoke test run separately for that.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch

import pytest

from classifier.nemotron import NemotronVisionModel
from prompts import PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT, render_product_match_verification_prompt

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.jpg"
    path.write_bytes(b"not-a-real-jpeg-but-non-empty")
    return path


def _model() -> NemotronVisionModel:
    return NemotronVisionModel(api_key="test-key")


# --- prompt rendering (already covered more thoroughly in prompts.py itself,
#     these confirm the model-facing message assembly uses it correctly) ----


def test_build_match_messages_includes_candidate_title_when_provided() -> None:
    model = _model()
    messages = model._build_match_messages(
        "base64data", "image/jpeg", "Kohinoor Charminar Sel Bsmti Rice 1KG PP", "Charminar Long Grain, 1kg"
    )
    user_text = messages[1]["content"][0]["text"]
    assert "Charminar Long Grain, 1kg" in user_text
    assert "Kohinoor Charminar Sel Bsmti Rice 1KG PP" in user_text


def test_build_match_messages_omits_candidate_title_when_none() -> None:
    model = _model()
    messages = model._build_match_messages("base64data", "image/jpeg", "KISSAN KETCHUP 950g", None)
    user_text = messages[1]["content"][0]["text"]
    assert "listing title" not in user_text.lower()


# --- verify_product_match end-to-end plumbing (mocked model call) ----------


def test_verify_product_match_passes_candidate_title_to_the_model_call(image_path: Path) -> None:
    model = _model()
    captured_messages: list[dict[str, Any]] = []

    async def fake_classify(messages: list[dict[str, Any]]) -> str:
        captured_messages.extend(messages)
        return json.dumps(
            {
                "is_match": False,
                "confidence": 0.9,
                "reason": "Different rice variant per listing title",
                "brand": "Kohinoor",
                "product_name": "Charminar Long Grain",
                "weight": "1 kg",
            }
        )

    with patch.object(NemotronVisionModel, "_classify", side_effect=fake_classify):
        verdict = _run(
            model.verify_product_match(
                image_path,
                "Kohinoor Charminar Sel Bsmti Rice 1KG PP",
                candidate_title="Charminar Long Grain, 1kg",
            )
        )

    assert verdict.is_match is False
    user_text = captured_messages[1]["content"][0]["text"]
    assert "Charminar Long Grain, 1kg" in user_text


def test_verify_product_match_defaults_candidate_title_to_none(image_path: Path) -> None:
    model = _model()
    captured_messages: list[dict[str, Any]] = []

    async def fake_classify(messages: list[dict[str, Any]]) -> str:
        captured_messages.extend(messages)
        return json.dumps(
            {"is_match": True, "confidence": 0.95, "reason": "ok", "brand": None, "product_name": None, "weight": None}
        )

    with patch.object(NemotronVisionModel, "_classify", side_effect=fake_classify):
        _run(model.verify_product_match(image_path, "KISSAN KETCHUP 950g"))

    user_text = captured_messages[1]["content"][0]["text"]
    assert "listing title" not in user_text.lower()


# --- system prompt content ---------------------------------------------------


def test_system_prompt_treats_differing_listing_title_as_meaningful_mismatch() -> None:
    prompt = PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT
    assert "listing title" in prompt.lower()
    assert "naming convention" in prompt.lower()
    # The exact worked example anchoring this fix - must instruct rejection.
    assert "Select Basmati" in prompt
    assert "Long Grain" in prompt
    assert "REJECTED" in prompt


def test_render_prompt_matches_what_build_match_messages_sends() -> None:
    rendered = render_product_match_verification_prompt(
        "Kohinoor Charminar Sel Bsmti Rice 1KG PP", "Charminar Long Grain, 1kg"
    )
    model = _model()
    messages = model._build_match_messages(
        "base64data", "image/jpeg", "Kohinoor Charminar Sel Bsmti Rice 1KG PP", "Charminar Long Grain, 1kg"
    )
    assert messages[1]["content"][0]["text"] == rendered
