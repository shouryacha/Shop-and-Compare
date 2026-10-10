"""
Nemotron 3 Nano Omni Vision-Language classifier adapter, served via NVIDIA Build.

Client for `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` hosted on NVIDIA
Build (build.nvidia.com), which exposes an OpenAI-compatible chat-completions
API - the same one the previous Qwen (and briefly-attempted Mistral)
adapters used (`https://integrate.api.nvidia.com/v1/chat/completions`).

WHY THIS MODEL (this session): Qwen (`qwen/qwen3.5-122b-a10b`) was
decommissioned by NVIDIA (confirmed live: HTTP 410 "has reached its end
of life"). A first migration to `mistralai/mistral-medium-3.5-128b` was
implemented per its official docs, but that model never returned a single
response across two full sessions of live testing (text-only, image,
streaming, with/without reasoning_effort, timeouts up to 280s) - while the
identical API key worked instantly against a different NVIDIA Build model
in the same session, ruling out an account/network problem. NVIDIA's own
API reference for Mistral Medium 3.5 notes it is governed by "NVIDIA API
Trial Terms of Service" (trial-tier only), which likely explains this.

Given that failure, this session searched NVIDIA Build's live model
catalog (build.nvidia.com/models, filtered toward Image-to-Text/VLM
models) rather than picking another single model by name, and compared
real candidates found there:
    - **nvidia/nemotron-3-nano-omni-30b-a3b-reasoning** (chosen): NVIDIA
      first-party, 33B (3B active MoE), explicitly tagged "OCR" and "VLM"
      on its own model card, has BOTH Free Endpoint AND Partner Endpoint
      availability (unlike Mistral's trial-only status), and shows healthy
      real usage (8M API calls in the last 30 days at the time of
      selection). Being NVIDIA's own model, not a third-party partner's,
      is the strongest long-term-availability signal available - the
      exact kind of risk (a partner's business decision to retire a
      model) that caused Qwen's removal.
    - minimaxai/minimax-m3: 427B params, explicitly labeled "Preview" in
      its own description - a weaker stability signal - and no OCR tag.
    - moonshotai/kimi-k2: 1T params, Free Endpoint only (no Partner
      Endpoint), no OCR tag, much larger/costlier per call for no
      demonstrated benefit to this project's use case.
Verified live before integration (not assumed): a text-only request, an
image request against a real product photo (correctly identified brand,
variant, and read on-pack text), a 2-image and a 5-image request (the
exact count `select_best_candidate` sends per side) all succeeded, and
the actual production verifier prompt/JSON schema was tested against real
matching and non-matching product images with correct, valid-JSON verdicts.

API DIFFERENCES FROM QWEN, verified live and against NVIDIA's official
model card (build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)
rather than assumed:
    - Endpoint and the image-embedding message shape
      (`{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<...>"}}`)
      are IDENTICAL to Qwen's. No message-building code needed structural
      changes.
    - Reasoning control is THE SAME mechanism Qwen used:
      `"chat_template_kwargs": {"enable_thinking": false}` as a top-level
      request field - confirmed live (an `extra_body`-wrapped form, as
      shown in this model's *self-hosted* NIM container documentation,
      was tried first and rejected by the hosted API with HTTP 400
      "Unsupported parameter(s): `extra_body`" - that wrapping is an
      OpenAI-SDK-level convenience, not a real field in the raw JSON body
      NVIDIA Build's hosted endpoint expects). Without disabling thinking,
      the model still returns a well-formed, separate `reasoning_content`
      field alongside a clean `message.content` (unlike Qwen, which could
      leave `message.content` empty if reasoning consumed the whole
      `max_tokens` budget) - `enable_thinking: false` is kept anyway, both
      to keep responses fast/cheap and as the same defensive posture Qwen
      required.
    - Max images per request: NVIDIA's docs did not state an explicit
      number for this model (unlike Qwen's NIM-for-VLM docs, which
      documented a default of 5). Verified live instead: a request with 5
      real images (the exact count `select_best_candidate` sends per
      side) succeeded and the model correctly reported the count back -
      `_MAX_IMAGES_PER_REQUEST` is kept at 5 on this empirical basis, not
      copied from Qwen's documentation.
    - Verifier prompt (`prompts.PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT`)
      was **not** changed - nothing about this model's behavior required a
      prompt adjustment, and the task asked to keep it unless required.

KNOWN BEHAVIORAL DIFFERENCE, reported honestly rather than hidden: in one
real test, this model accepted a Tandoori Masala flavor variant against
the generic ERP name "FUNFOODS MAYO COMBO 100g" (`is_match: true`),
where Qwen previously rejected the same case as a flavor mismatch. The
ERP name does not explicitly specify a flavor, so this is a genuinely
defensible edge-case judgment call rather than a broken verifier - but it
means flavor-unspecified generic ERP names may accept a named-flavor
variant more readily than Qwen did. Unambiguous cases (correct product
accepted, entirely wrong product rejected) both behaved correctly.

Two independent public entry points:
    - `select_best_candidate()`  — the comparative workflow: given several
                                    candidate images for the same product/
                                    side, sends them ALL in a single request
                                    and asks the model to rank and pick the
                                    one best match, rather than scoring each
                                    image independently. This is what
                                    `selection_stage.py` uses instead of a
                                    separate selector.
    - `verify_product_match()`   — a single yes/no question ("is this
                                    retailer page's product the one that was
                                    requested?") over one image, returning
                                    `models.ProductMatchVerdict` rather than
                                    `ClassificationVerdict` - a deliberately
                                    separate model, since this judgement has
                                    no notion of image "side" at all. Used by
                                    `retailer_gallery_stage.py` before
                                    committing to a retailer's full gallery.

(A third entry point, per-image `classify_image()`/`classify()`, existed
here previously but was removed - it had no callers anywhere in the
project; both real orchestrators above always used the comparative or
match-verification path instead.)

Both entry points share the same low-level request machinery
(`_classify()` → `_post_chat_completion()`), just with different message
builders (`_build_comparative_messages()` / `_build_match_messages()`) and
response parsers (`_parse_comparative_response()` / `_parse_match_response()`).
`_classify()` also retries transient NVIDIA Build failures (HTTP 500s,
malformed/missing/empty responses, timeouts) a small number of times before
raising - these are API failures, not a verdict, so a caller catching the
eventual exception must not treat it as "rejected" (see
`retailer_gallery_stage.py`).

`models.ClassificationVerdict` (still used as `select_best_candidate()`'s
return type) has no dedicated brand/product_name/weight fields, so the full
structured JSON payload the model returns - which includes those fields
whenever the model can read them off the packaging - is preserved on
`ClassificationVerdict.raw_model_response` rather than being discarded.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from config import Settings, get_settings
from models import (
    ClassificationVerdict,
    ImageSide,
    ProductMatchVerdict,
    ShoppingMatchClassification,
    ShoppingMatchVerdict,
)
from prompts import (
    PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT,
    render_comparative_selection_system_prompt,
    render_comparative_selection_user_prompt,
    render_product_match_verification_prompt,
    render_shopping_offer_classification_system_prompt,
    render_shopping_offer_classification_user_prompt,
)

logger = logging.getLogger(__name__)

_MODEL_NAME: Final[str] = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
_PROVIDER_KEY: Final[str] = "nemotron_vl"
_DEFAULT_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
_CHAT_COMPLETIONS_PATH: Final[str] = "/chat/completions"

# The API key itself is read from `config.Settings.nvidia_api_key` (see
# config.py); this constant is kept only for the error message below, since
# that's the env var / `.env` entry a user would actually set.
_API_KEY_ENV_VAR: Final[str] = "NVIDIA_API_KEY"
_BASE_URL_ENV_VAR: Final[str] = "NVIDIA_BUILD_BASE_URL"

# A vision-language completion is slower than a typical HTTP call; use this
# as a floor even if `config.Settings.request_timeout_seconds` is configured
# lower for other, non-VLM providers.
_MIN_TIMEOUT_SECONDS: Final[float] = 60.0

_TEMPERATURE: Final[float] = 0.0  # deterministic classification, not creative generation
_MAX_TOKENS: Final[int] = 500

# Shopping classification answers for up to 5 candidates in one response
# (label, confidence, reason, evidence and four true/false fields each). A
# full live response was about 1,900 characters - right at the 500-token
# default, where truncation would make the JSON unusable. Only this request
# type uses the larger limit; every other call keeps `_MAX_TOKENS`.
_SHOPPING_CLASSIFICATION_MAX_TOKENS: Final[int] = 800

# NVIDIA Build occasionally fails transiently (HTTP 500s, malformed JSON,
# missing/empty message.content - all observed in real logs) that have
# nothing to do with whether a candidate is the right product. These are
# retried a small number of times before giving up, so a single flaky
# response doesn't eliminate a candidate the model was never actually
# asked to judge. A bad API key (401) is NOT retried - retrying can't fix
# that, and would just waste `_MAX_VISION_RETRIES` round-trips.
_MAX_VISION_RETRIES: Final[int] = 2  # up to 3 attempts total (1 + 2 retries)
_VISION_RETRY_BACKOFF_SECONDS: Final[float] = 2.0

# Verified LIVE this session with a real 5-image request (the exact count
# `select_best_candidate` sends per side) - the model correctly reported
# "5" back. NVIDIA's docs did not state an explicit max for this model, so
# this is an empirical finding, not a copied/assumed number.
_MAX_IMAGES_PER_REQUEST: Final[int] = 5

_SUPPORTED_IMAGE_MIME_TYPES: Final[dict[str, str]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Same mechanism Qwen used - verified live this session that the hosted
# NVIDIA Build endpoint for THIS model accepts this exact top-level field
# (an `extra_body`-wrapped form, shown in the model's self-hosted-NIM
# docs, was tried first and rejected with HTTP 400 "Unsupported
# parameter(s): `extra_body`" - see module docstring).
_CHAT_TEMPLATE_KWARGS: Final[dict[str, bool]] = {"enable_thinking": False}

# The prompt shows the model the label "not_relevant" for
# `ShoppingMatchClassification.DIFFERENT_VARIANT`. In live runs the model
# kept choosing the label "different_variant" for any candidate it described
# as "a different variant" of the requested product - which should be
# "related_product" - because the label's name matched its own wording. The
# category itself and the enum are unchanged; only the wording the model
# sees differs. The enum's own value is still accepted.
_SHOPPING_LABEL_ALIASES: Final[dict[str, str]] = {
    "not_relevant": ShoppingMatchClassification.DIFFERENT_VARIANT.value,
}

# The required fields of one shopping-classification entry, in the order the
# prompt's schema lists them (see `_salvage_shopping_classifications`).
_SHOPPING_ENTRY_REQUIRED_FIELDS: Final[re.Pattern[str]] = re.compile(
    r'"image_index"\s*:\s*(?P<index>\d+)\s*,\s*'
    r'"classification"\s*:\s*"(?P<classification>[A-Za-z_]+)"\s*,\s*'
    r'"confidence"\s*:\s*(?P<confidence>\d+(?:\.\d+)?)'
    r'(?:\s*,\s*"reason"\s*:\s*"(?P<reason>(?:[^"\\]|\\.)*)")?'
)


class VisionModelError(Exception):
    """Raised when the vision model cannot produce a usable verdict."""


class VisionModelAuthError(VisionModelError):
    """Raised when NVIDIA Build rejects the configured API key."""


class VisionModelRateLimitError(VisionModelError):
    """Raised when NVIDIA Build reports the account's rate/usage limit was hit."""


class VisionModelTimeoutError(VisionModelError):
    """Raised when the request does not complete within the configured timeout."""


class NemotronVisionModel:
    """
    Nemotron 3 Nano Omni (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`)
    client via NVIDIA Build. Deliberately not behind an abstract port: both
    capabilities below (`select_best_candidate`, `verify_product_match`) are
    specific to this adapter, and every caller depends on this concrete
    class directly. See `interfaces.py`'s module docstring for why the
    former `VisionModel` port was removed rather than kept on spec.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Args:
            settings: Injected config; defaults to `config.get_settings()`.
            api_key: Explicit override, mainly for tests. Falls back to
                `Settings.nvidia_api_key` (populated from the `NVIDIA_API_KEY`
                environment variable / `.env` entry - see `config.py`).
            base_url: Explicit override, mainly for tests. Falls back to the
                `NVIDIA_BUILD_BASE_URL` environment variable, then the
                documented NVIDIA Build default.

        Raises:
            VisionModelError: if no API key can be resolved from any source.
        """
        self._settings = settings or get_settings()

        resolved_api_key = api_key or self._settings.nvidia_api_key
        if not resolved_api_key:
            raise VisionModelError(
                "No NVIDIA Build API key configured. Set the "
                f"{_API_KEY_ENV_VAR} environment variable."
            )
        self._api_key = resolved_api_key
        self._base_url = base_url or os.environ.get(_BASE_URL_ENV_VAR, _DEFAULT_BASE_URL)
        self._timeout_seconds = max(self._settings.request_timeout_seconds, _MIN_TIMEOUT_SECONDS)

    @property
    def name(self) -> str:
        return _PROVIDER_KEY

    async def select_best_candidate(
        self, image_paths: list[Path], product_name: str, side: ImageSide
    ) -> tuple[int, ClassificationVerdict]:
        """
        Compare all of `image_paths` together in a single request and return
        the 0-based index (into `image_paths`) of the single best match,
        together with a `ClassificationVerdict` describing it.

        Unlike `classify_image`, this never judges an image in isolation:
        every candidate is sent in one request so the model ranks them
        against each other directly, per this model's verified multi-image
        support (`_MAX_IMAGES_PER_REQUEST`).

        Raises:
            ValueError: if `image_paths` is empty or exceeds
                `_MAX_IMAGES_PER_REQUEST`.
            VisionModelError: if any image cannot be read, the request to
                NVIDIA Build fails, or the response cannot be parsed into a
                valid selection.
        """
        if not image_paths:
            raise ValueError("image_paths must be non-empty")
        if len(image_paths) > _MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"select_best_candidate got {len(image_paths)} images, but NVIDIA Build "
                f"supports at most {_MAX_IMAGES_PER_REQUEST} per request"
            )

        logger.info(
            "Comparing %d candidate(s) for product=%r side=%s using model=%s",
            len(image_paths),
            product_name,
            side.value,
            _MODEL_NAME,
        )

        try:
            loaded_images = [self._load_image(path) for path in image_paths]
            messages = self._build_comparative_messages(loaded_images, product_name, side)
            raw_response = await self._classify(messages)
            zero_based_index, verdict = self._parse_comparative_response(
                raw_response, len(image_paths), side
            )
        except VisionModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a domain error below
            logger.exception("Unexpected failure comparing candidate images")
            raise VisionModelError(f"Failed to compare candidate images: {exc}") from exc

        logger.info(
            "Selected candidate index=%d (0-based) confidence=%.2f for side=%s",
            zero_based_index,
            verdict.confidence,
            side.value,
        )
        return zero_based_index, verdict

    def _build_comparative_messages(
        self, loaded_images: list[tuple[str, str]], product_name: str, side: ImageSide
    ) -> list[dict[str, Any]]:
        """
        Assemble the OpenAI-compatible chat message list for comparative
        selection: a system prompt describing the priority-ordered criteria
        and required JSON schema (from `prompts.py`), plus a single user
        message carrying the product/side context followed by every
        candidate image, in the same order as `loaded_images`.
        """
        num_images = len(loaded_images)
        system_prompt = render_comparative_selection_system_prompt(side.value, num_images)
        user_text = render_comparative_selection_user_prompt(product_name, side.value, num_images)

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_b64, mime_type in loaded_images:
            data_uri = f"data:{mime_type};base64,{image_b64}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _parse_comparative_response(
        self, raw_response: str, num_images: int, side: ImageSide
    ) -> tuple[int, ClassificationVerdict]:
        """
        Parse the model's raw text into `(zero_based_index, ClassificationVerdict)`.

        `is_correct_product` is set to `True` on the returned verdict: the
        comparative prompt already ranks "correct product" as its top
        priority criterion, so being selected at all presumes the model
        judged it a match - there is no separate reject/none-of-these
        signal in this response schema.

        Raises:
            VisionModelError: if the response is not valid JSON, is missing
                required fields, has an out-of-range `best_image_index`, or
                fails `ClassificationVerdict` validation.
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise VisionModelError(f"Model response was not valid JSON: {raw_response!r}") from exc

        if not isinstance(payload, dict):
            raise VisionModelError(f"Model response JSON was not an object: {raw_response!r}")

        try:
            one_based_index = int(payload["best_image_index"])
        except (KeyError, ValueError, TypeError) as exc:
            raise VisionModelError(
                f"Model response has missing/invalid 'best_image_index': {payload!r}"
            ) from exc

        if not (1 <= one_based_index <= num_images):
            raise VisionModelError(
                f"Model returned best_image_index={one_based_index}, out of range "
                f"1..{num_images}: {payload!r}"
            )

        try:
            verdict = ClassificationVerdict(
                is_correct_product=True,
                side=side,
                confidence=float(payload["confidence"]),
                reasoning=str(payload.get("reason", "")),
                raw_model_response=payload,
                model_name=_PROVIDER_KEY,
            )
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            raise VisionModelError(
                f"Model response missing/invalid required fields: {payload!r}"
            ) from exc

        return one_based_index - 1, verdict

    async def verify_product_match(
        self, image_path: Path, product_name: str, candidate_title: str | None = None
    ) -> ProductMatchVerdict:
        """
        Judge whether `image_path` (typically a retailer product page's
        first/primary gallery image) is clearly the same commercial product
        as `product_name`, tolerating retailer wording differences but
        rejecting a genuinely different product family or variant.

        `candidate_title`: the retailer's own listing title for the exact
        product in `image_path`, when the caller has one available (e.g.
        SerpApi's Amazon Product API `product_results.title` - see
        `search/retailers/amazon_serpapi.py`). Passed to the model as
        additional textual evidence alongside the image, never as a
        replacement for it - added after a live false-positive where a
        genuinely different product variant ("Long Grain" vs the requested
        "Select Basmati") was accepted as just a naming-convention
        difference, because the model had no visibility into what the
        retailer itself calls the pictured product. `None` (the default)
        behaves exactly as before this parameter existed.

        Distinct responsibility from `classify_image`: this never judges
        `side` (front/back) - it answers the prior question of whether a
        retailer page should be trusted at all, via `models.ProductMatchVerdict`
        rather than `models.ClassificationVerdict`.

        Raises:
            VisionModelError: if the image cannot be read, the request to
                NVIDIA Build fails, or the response cannot be parsed into a
                valid `ProductMatchVerdict`.
        """
        logger.info(
            "Verifying product match for image=%s product=%r candidate_title=%r using model=%s",
            image_path,
            product_name,
            candidate_title,
            _MODEL_NAME,
        )

        try:
            image_b64, mime_type = self._load_image(image_path)
            messages = self._build_match_messages(image_b64, mime_type, product_name, candidate_title)
            raw_response = await self._classify(messages)
            verdict = self._parse_match_response(raw_response)
        except VisionModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a domain error below
            logger.exception("Unexpected failure verifying product match for image=%s", image_path)
            raise VisionModelError(f"Failed to verify product match for {image_path}: {exc}") from exc

        logger.info(
            "Verified image=%s: is_match=%s confidence=%.2f",
            image_path,
            verdict.is_match,
            verdict.confidence,
        )
        return verdict

    def _build_match_messages(
        self, image_b64: str, mime_type: str, product_name: str, candidate_title: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Assemble the OpenAI-compatible chat message list for product-match
        verification: the system prompt describing the comparison order and
        tolerance rules (from `prompts.py`), plus a user message carrying
        the requested product name, the retailer's own listing title when
        available, and the single candidate image.
        """
        data_uri = f"data:{mime_type};base64,{image_b64}"
        return [
            {"role": "system", "content": PRODUCT_MATCH_VERIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": render_product_match_verification_prompt(product_name, candidate_title),
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]

    def _parse_match_response(self, raw_response: str) -> ProductMatchVerdict:
        """
        Parse the model's raw text into a validated `ProductMatchVerdict`.

        Raises:
            VisionModelError: if the response is not valid JSON, is missing
                required fields, or fails `ProductMatchVerdict` validation.
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise VisionModelError(f"Model response was not valid JSON: {raw_response!r}") from exc

        if not isinstance(payload, dict):
            raise VisionModelError(f"Model response JSON was not an object: {raw_response!r}")

        try:
            return ProductMatchVerdict(
                is_match=bool(payload["is_match"]),
                confidence=float(payload["confidence"]),
                reason=str(payload.get("reason", "")),
                brand=payload.get("brand"),
                product_name=payload.get("product_name"),
                weight=payload.get("weight"),
                raw_model_response=payload,
                model_name=_PROVIDER_KEY,
            )
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            raise VisionModelError(
                f"Model response missing/invalid required fields: {payload!r}"
            ) from exc

    async def classify_shopping_offers(
        self,
        image_paths: list[Path | None],
        product_name: str,
        offer_labels: list[str],
        canonical_title: str | None = None,
    ) -> list[ShoppingMatchVerdict]:
        """
        Classify every candidate INDEPENDENTLY against `product_name`, in a
        single batched request, for the shopping-comparison feature
        (`shopping_comparison_stage.py`). Unlike `select_best_candidate`,
        this never picks a single winner - every candidate gets its own
        `ShoppingMatchVerdict`
        (`SAME_PRODUCT` / `SAME_PRODUCT_DIFFERENT_SIZE` / `RELATED_PRODUCT` / `DIFFERENT_VARIANT`),
        returned in the same order as `image_paths`, because a shopping
        comparison is expected to be a genuine mix of listings, not
        candidates for one slot.

        There is one entry per candidate in both lists. `offer_labels[i]`
        is that candidate's TEXT evidence - listing title, retailer, price,
        URL and product-page text when available - which the model treats
        as primary. `image_paths[i]` is its supplemental image, or `None`
        when there isn't one; such a candidate is classified from its text
        alone. `canonical_title` is an optional verified reference title.

        Raises:
            ValueError: if `image_paths` is empty, exceeds
                `_MAX_IMAGES_PER_REQUEST`, or `offer_labels` isn't the same
                length as `image_paths`.
            VisionModelError: if any image cannot be read, the request to
                NVIDIA Build fails, or the response cannot be parsed into a
                valid, complete set of verdicts.
        """
        if not image_paths:
            raise ValueError("image_paths must be non-empty")
        if len(image_paths) > _MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"classify_shopping_offers got {len(image_paths)} images, but NVIDIA Build "
                f"supports at most {_MAX_IMAGES_PER_REQUEST} per request"
            )
        if len(offer_labels) != len(image_paths):
            raise ValueError(
                f"offer_labels ({len(offer_labels)}) must be the same length as "
                f"image_paths ({len(image_paths)})"
            )

        logger.info(
            "Classifying %d shopping offer(s) for product=%r using model=%s",
            len(image_paths),
            product_name,
            _MODEL_NAME,
        )

        try:
            loaded_images = [self._load_image(path) if path is not None else None for path in image_paths]
            messages = self._build_shopping_classification_messages(
                loaded_images, product_name, offer_labels, canonical_title
            )
            raw_response = await self._classify(messages, max_tokens=_SHOPPING_CLASSIFICATION_MAX_TOKENS)
            verdicts = self._parse_shopping_classification_response(raw_response, len(image_paths))
        except VisionModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a domain error below
            logger.exception("Unexpected failure classifying shopping offers")
            raise VisionModelError(f"Failed to classify shopping offers: {exc}") from exc

        logger.info(
            "Classified %d shopping offer(s): %s",
            len(verdicts),
            [v.classification.value for v in verdicts],
        )
        return verdicts

    def _build_shopping_classification_messages(
        self,
        loaded_images: list[tuple[str, str] | None],
        product_name: str,
        offer_labels: list[str],
        canonical_title: str | None,
    ) -> list[dict[str, Any]]:
        """
        Assemble the chat message list for batched shopping-offer
        classification (see prompts.py): one text block describing every
        candidate, then the available images in candidate order. Each
        candidate's text says which attached image is its own, or that it
        has none - so a missing image never shifts the others.
        """
        attached = [image for image in loaded_images if image is not None]
        labels_with_image_notes: list[str] = []
        attached_number = 0
        for label, image in zip(offer_labels, loaded_images):
            if image is None:
                note = "Image: not available - judge this candidate from its text"
            else:
                attached_number += 1
                note = f"Image: attached image {attached_number} of {len(attached)}"
            labels_with_image_notes.append(f"{label}\n   {note}")

        system_prompt = render_shopping_offer_classification_system_prompt(len(loaded_images))
        user_text = render_shopping_offer_classification_user_prompt(
            product_name, labels_with_image_notes, canonical_title
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_b64, mime_type in attached:
            data_uri = f"data:{mime_type};base64,{image_b64}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _parse_shopping_classification_response(
        self, raw_response: str, num_images: int
    ) -> list[ShoppingMatchVerdict]:
        """
        Parse the model's raw text into `num_images` `ShoppingMatchVerdict`s,
        reordered into `image_index` order (1-based in the response, 0-based
        position in the returned list).

        Raises:
            VisionModelError: if the response is not valid JSON, is missing
                required fields, has an out-of-range/duplicate/missing
                `image_index`, or any entry fails `ShoppingMatchVerdict`
                validation.
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            payload = self._salvage_shopping_classifications(cleaned)
            if payload is None:
                raise VisionModelError(f"Model response was not valid JSON: {raw_response!r}") from exc
            logger.warning(
                "Shopping classification response was not valid JSON (%s); recovered %d entr(y/ies) "
                "from its required fields",
                exc,
                len(payload["classifications"]),
            )

        if not isinstance(payload, dict) or not isinstance(payload.get("classifications"), list):
            raise VisionModelError(f"Model response JSON missing 'classifications' list: {raw_response!r}")

        by_index: dict[int, ShoppingMatchVerdict] = {}
        for entry in payload["classifications"]:
            try:
                image_index = int(entry["image_index"])
                verdict = ShoppingMatchVerdict(
                    classification=self._parse_shopping_label(entry["classification"]),
                    confidence=float(entry["confidence"]),
                    reason=str(entry.get("reason", "")),
                    evidence=self._parse_evidence(entry.get("evidence")),
                    brand_match=self._parse_optional_bool(entry.get("brand_match")),
                    product_match=self._parse_optional_bool(entry.get("product_match")),
                    variant_match=self._parse_optional_bool(entry.get("variant_match")),
                    size_match=self._parse_optional_bool(entry.get("size_match")),
                    model_name=_PROVIDER_KEY,
                )
            except (KeyError, ValueError, TypeError, ValidationError) as exc:
                raise VisionModelError(
                    f"Model response has an invalid classification entry: {entry!r}"
                ) from exc

            if not (1 <= image_index <= num_images):
                raise VisionModelError(
                    f"Model returned image_index={image_index}, out of range 1..{num_images}: {payload!r}"
                )
            if image_index in by_index:
                raise VisionModelError(f"Model returned duplicate image_index={image_index}: {payload!r}")
            by_index[image_index] = verdict

        if len(by_index) != num_images:
            missing = sorted(set(range(1, num_images + 1)) - by_index.keys())
            raise VisionModelError(f"Model response is missing classification(s) for image_index={missing}")

        return [by_index[i] for i in range(1, num_images + 1)]

    @staticmethod
    def _parse_shopping_label(raw: Any) -> ShoppingMatchClassification:
        """
        Map the label the model returned to a `ShoppingMatchClassification`.
        Raises `ValueError` for an unknown label (the caller turns that into
        a `VisionModelError`).
        """
        label = str(raw).strip().lower()
        return ShoppingMatchClassification(_SHOPPING_LABEL_ALIASES.get(label, label))

    @staticmethod
    def _salvage_shopping_classifications(text: str) -> dict[str, Any] | None:
        """
        Recover the REQUIRED fields of each entry from a response that is
        not valid JSON. Live responses have twice been broken only inside
        the optional true/false fields (a stray comma; a duplicated key),
        which cost a whole batch of otherwise correct classifications.

        Returns a payload in the normal shape, with only `image_index`,
        `classification`, `confidence` and `reason` per entry - or `None`
        if no entry can be read. The caller still runs every normal check
        (valid category, index range, no duplicates, none missing), so a
        response that is broken beyond its optional fields is still rejected.
        """
        entries = [
            {
                "image_index": match["index"],
                "classification": match["classification"],
                "confidence": match["confidence"],
                "reason": match["reason"] or "",
            }
            for match in _SHOPPING_ENTRY_REQUIRED_FIELDS.finditer(text)
        ]
        return {"classifications": entries} if entries else None

    @staticmethod
    def _parse_evidence(raw: Any) -> list[str]:
        """
        Best-effort extraction of the optional `evidence` list. Unlike the
        required fields above (classification/confidence, which still fail
        the whole entry if malformed), a malformed `evidence` value must
        never take down an otherwise-valid classification - it degrades to
        an empty list instead (see `models.ShoppingMatchVerdict`'s
        docstring on these fields being additive/optional).
        """
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw if isinstance(item, (str, int, float))]

    @staticmethod
    def _parse_optional_bool(raw: Any) -> bool | None:
        """Best-effort extraction of an optional boolean attribute flag - `None` for anything else, never raises."""
        return raw if isinstance(raw, bool) else None

    def _load_image(self, image_path: Path) -> tuple[str, str]:
        """
        Read `image_path` from disk and return `(base64_encoded_str, mime_type)`.

        Raises:
            VisionModelError: if the path is missing, unreadable, empty, or
                has an unsupported extension.
        """
        if not image_path.is_file():
            raise VisionModelError(f"Image path does not exist or is not a file: {image_path}")

        mime_type = _SUPPORTED_IMAGE_MIME_TYPES.get(image_path.suffix.lower())
        if mime_type is None:
            raise VisionModelError(
                f"Unsupported image extension '{image_path.suffix}' for {image_path}; "
                f"supported extensions: {sorted(_SUPPORTED_IMAGE_MIME_TYPES)}"
            )

        try:
            raw_bytes = image_path.read_bytes()
        except OSError as exc:
            raise VisionModelError(f"Could not read image file {image_path}: {exc}") from exc

        if not raw_bytes:
            raise VisionModelError(f"Image file is empty: {image_path}")

        return base64.b64encode(raw_bytes).decode("ascii"), mime_type

    async def _classify(self, messages: list[dict[str, Any]], max_tokens: int = _MAX_TOKENS) -> str:
        """
        Send `messages` to NVIDIA Build's OpenAI-compatible chat-completions
        endpoint for `_MODEL_NAME` and return the assistant's raw text
        response.

        Transient failures (HTTP 500s, malformed JSON, missing/empty
        `message.content`, timeouts, network errors) are retried up to
        `_MAX_VISION_RETRIES` times with a short backoff before this raises
        - these are API failures, not a verdict, so callers (see
        `retailer_gallery_stage.py`) must not treat an exception from here
        as "the product doesn't match." A bad API key
        (`VisionModelAuthError`) is never retried, since no amount of
        retrying fixes an invalid credential.

        Raises:
            VisionModelAuthError: if NVIDIA Build rejects the API key.
            VisionModelRateLimitError: if the account's rate/usage limit was
                hit and retries were exhausted.
            VisionModelTimeoutError: if the request did not complete within
                `self._timeout_seconds` and retries were exhausted.
            VisionModelError: for any other request/network/response
                failure, after retries were exhausted.
        """
        url = f"{self._base_url.rstrip('/')}{_CHAT_COMPLETIONS_PATH}"
        body = {
            "model": _MODEL_NAME,
            "messages": messages,
            "temperature": _TEMPERATURE,
            "max_tokens": max_tokens,
            "chat_template_kwargs": _CHAT_TEMPLATE_KWARGS,
        }

        last_error: VisionModelError | None = None
        max_attempts = _MAX_VISION_RETRIES + 1
        for attempt in range(1, max_attempts + 1):
            try:
                payload = await asyncio.to_thread(self._post_chat_completion, url, body)
                return self._extract_message_content(payload)
            except VisionModelAuthError:
                raise  # never retry - a bad API key won't become valid
            except VisionModelError as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - translated to a domain error
                logger.exception("Unexpected failure calling NVIDIA Build")
                last_error = VisionModelError(f"Unexpected failure calling NVIDIA Build: {exc}")

            if attempt < max_attempts:
                logger.warning(
                    "NVIDIA Build call failed (attempt %d/%d): %s - retrying",
                    attempt, max_attempts, last_error,
                )
                await asyncio.sleep(_VISION_RETRY_BACKOFF_SECONDS)

        assert last_error is not None
        raise last_error

    def _log_request_debug(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> None:
        """
        DEBUG-level visibility into exactly what's about to be sent to NVIDIA
        Build, with the API key masked and any base64 image data truncated -
        useful when diagnosing an unexpected response or timeout. Guarded by
        `isEnabledFor` so the (moderately expensive) deep-copy/truncation/
        JSON-dump work is skipped entirely unless DEBUG logging is enabled.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Base URL: %s\nFull request URL: %s\nModel name: %s\nHTTP headers: %s\nJSON request body:\n%s",
            self._base_url,
            url,
            body.get("model"),
            self._mask_headers(headers),
            json.dumps(self._truncate_body_for_debug(body), indent=2),
        )

    @staticmethod
    def _mask_headers(headers: dict[str, str]) -> dict[str, str]:
        """Return a copy of `headers` with the Authorization bearer token masked."""
        masked = dict(headers)
        auth = masked.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth[len("Bearer "):]
            visible = token[:4]
            masked["Authorization"] = f"Bearer {visible}{'*' * max(len(token) - 4, 0)}"
        return masked

    @staticmethod
    def _truncate_body_for_debug(
        body: dict[str, Any], max_data_uri_chars: int = 60
    ) -> dict[str, Any]:
        """Return a deep copy of `body` with any base64 image data URI truncated for printing."""
        truncated = copy.deepcopy(body)
        for message in truncated.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not (isinstance(part, dict) and part.get("type") == "image_url"):
                    continue
                image_url = part.get("image_url", {})
                url_value = image_url.get("url", "")
                if isinstance(url_value, str) and len(url_value) > max_data_uri_chars:
                    image_url["url"] = (
                        f"{url_value[:max_data_uri_chars]}...<truncated, "
                        f"{len(url_value)} chars total>"
                    )
        return truncated

    def _post_chat_completion(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        Blocking HTTP POST + JSON parse (invoked via `asyncio.to_thread`).
        Isolated into its own method so the sync/async boundary lives in
        exactly one place.
        """
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        request = urllib.request.Request(url, data=data, method="POST", headers=headers)
        self._log_request_debug(url, body, headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            self._raise_for_http_error(exc.code, exc.read())
            raise  # unreachable - _raise_for_http_error always raises
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise VisionModelTimeoutError(
                    f"Timed out calling NVIDIA Build after {self._timeout_seconds}s"
                ) from exc
            raise VisionModelError(f"Network error calling NVIDIA Build: {exc.reason}") from exc
        except TimeoutError as exc:
            raise VisionModelTimeoutError(
                f"Timed out calling NVIDIA Build after {self._timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise VisionModelError(f"Network error calling NVIDIA Build: {exc}") from exc

        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise VisionModelError(f"NVIDIA Build response was not valid JSON: {exc}") from exc

    def _raise_for_http_error(self, status_code: int, body: bytes) -> None:
        """Translate a non-2xx HTTP response into a specific domain exception."""
        message = self._extract_error_message(body) or f"HTTP {status_code}"

        if status_code == 401:
            raise VisionModelAuthError(f"NVIDIA Build rejected the API key: {message}")
        if status_code == 429:
            raise VisionModelRateLimitError(f"NVIDIA Build rate limit reached: {message}")
        raise VisionModelError(f"NVIDIA Build request failed with HTTP {status_code}: {message}")

    @staticmethod
    def _extract_error_message(body: bytes) -> str | None:
        """Best-effort extraction of an error message from an HTTP-error response body."""
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        error = parsed.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error) if error else None

    @staticmethod
    def _extract_message_content(payload: dict[str, Any]) -> str:
        """
        Pull the assistant's text out of an OpenAI-style chat-completions
        response. If `message.content` is missing or empty, the complete raw
        JSON response is logged at ERROR level for inspection before raising.
        """
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "NVIDIA Build response has no message.content. Full raw JSON response:\n%s",
                json.dumps(payload, indent=2),
            )
            raise VisionModelError(
                f"Unexpected NVIDIA Build response shape: {payload!r}"
            ) from exc

        if not content:
            logger.error(
                "NVIDIA Build returned an empty message.content. Full raw JSON response:\n%s",
                json.dumps(payload, indent=2),
            )
            raise VisionModelError("NVIDIA Build returned an empty message.content")

        return content
