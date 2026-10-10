"""
Deterministic post-validation checks applied AFTER the vision model's
verdict, never inside the verifier prompt itself.

Motivation (found live, see HANDOFF.md): the model sometimes correctly
extracts a `brand` that is clearly different from the requested ERP brand
(e.g. requested "JERSEY", extracted "Mother Dairy") but still returns
`is_match: true` - its own stated `reason` doesn't even engage with the
brand comparison in these cases. Reproduced with the exact prompt/response:
the prompt already explicitly requires strict brand rejection, and the
response parsing is a faithful, unmodified pass-through - so this is a
genuine model reasoning gap, not a prompt-wording or parsing bug. Rather
than continue tuning prompt wording, this module adds a small, deterministic
guardrail on top of the model's own already-extracted structured fields.

Design:
    - One-directional only: can flip `is_match: True -> False`, never the
      reverse. This is a stricter guardrail, not a second opinion that can
      overturn a correct rejection into a false accept.
    - Conservative by construction: only overrides when BOTH the ERP brand
      and the model's extracted brand are non-empty AND confidently judged
      to be different (see `_brands_equivalent`). Any ambiguity - either
      brand missing, or a plausible match - leaves the model's own verdict
      untouched.
    - `_CHECKS` is a plain ordered list of independent check functions, each
      with the same `(verdict, erp_brand) -> tuple[ProductMatchVerdict, OverrideInfo] | None`
      shape (`None` = no override). Adding a future check (e.g. weight/pack
      size) means writing one more such function and appending it here -
      nothing else in this module or its caller needs to change.
    - `OverrideInfo` carries everything a caller needs to log a fired
      override in structured form (which check fired, both brand strings,
      the model's own original is_match/confidence). This module never logs
      anything itself - it doesn't know the retailer/candidate/URL context
      that `retailer_gallery_stage.py` has; it just hands back the data for
      the caller to log.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from models import ProductMatchVerdict


@dataclass(frozen=True)
class OverrideInfo:
    """Everything a caller needs to log a fired deterministic override in structured form."""

    check_name: str
    erp_brand: str
    extracted_brand: str
    match_method: str
    original_is_match: bool
    original_confidence: float


# Compacted (see `_normalize_brand`) name groups that are the SAME real-world
# brand/manufacturer despite being written differently - reusing only the
# house-brand tolerances the verifier prompt itself already documents as
# intentional (see prompts.py's worked examples), not inventing new ones.
_BRAND_ALIAS_GROUPS: list[frozenset[str]] = [
    frozenset({"gemini", "tatatea", "brookebond"}),
    frozenset({"funfoods", "droetker", "droetkerfunfoods"}),
]

_FUZZY_MATCH_THRESHOLD = 0.75


def _normalize_brand(raw: str) -> str:
    """Lowercase, strip everything but letters/digits - so case, punctuation, and spacing never matter."""
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def _same_alias_group(a: str, b: str) -> bool:
    return any(a in group and b in group for group in _BRAND_ALIAS_GROUPS)


def _brands_equivalent(erp_brand: str, extracted_brand: str) -> bool:
    """
    Whether two brand strings should be treated as the SAME brand for
    override purposes. Deliberately permissive - this only gates whether an
    override is CONFIDENT enough to fire, so false negatives here (missing a
    real alias) just mean "defer to the AI", not "wrongly reject".
    """
    a, b = _normalize_brand(erp_brand), _normalize_brand(extracted_brand)
    if not a or not b:
        return True  # can't judge - don't treat as a confident mismatch
    if a == b:
        return True
    if a in b or b in a:
        return True
    if _same_alias_group(a, b):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= _FUZZY_MATCH_THRESHOLD


def _brand_mismatch_check(
    verdict: ProductMatchVerdict, erp_brand: str | None
) -> tuple[ProductMatchVerdict, OverrideInfo] | None:
    """
    Force `is_match: False` when the ERP brand and the model's own
    extracted brand are both non-empty and confidently different -
    regardless of what the model itself decided.
    """
    if not erp_brand or not erp_brand.strip():
        return None  # no reliable ERP-side brand to compare against
    if not verdict.brand or not verdict.brand.strip():
        return None  # model couldn't read a brand - nothing to compare

    if _brands_equivalent(erp_brand, verdict.brand):
        return None  # not confidently different - defer to the AI's own verdict

    overridden = verdict.model_copy(
        update={
            "is_match": False,
            "reason": (
                f"[deterministic brand-mismatch override] ERP brand {erp_brand!r} does not match "
                f"extracted brand {verdict.brand!r}; model's own reason was: {verdict.reason!r}"
            ),
        }
    )
    info = OverrideInfo(
        check_name="brand_mismatch",
        erp_brand=erp_brand,
        extracted_brand=verdict.brand,
        match_method="No equivalence",  # compact/substring/alias/fuzzy all failed - see _brands_equivalent
        original_is_match=verdict.is_match,
        original_confidence=verdict.confidence,
    )
    return overridden, info


_CHECKS = [_brand_mismatch_check]


def apply_deterministic_checks(
    verdict: ProductMatchVerdict, erp_brand: str | None
) -> tuple[ProductMatchVerdict, OverrideInfo | None]:
    """
    Run every registered deterministic check against `verdict` in order,
    returning `(verdict, None)` unchanged if none fire, or
    `(overridden_verdict, info)` for the first one that does. `erp_brand`
    is the ERP source's own brand field (e.g. an Excel "Brand" column) -
    `None` when no such field is available, in which case no check in this
    module can fire.
    """
    for check in _CHECKS:
        result = check(verdict, erp_brand)
        if result is not None:
            return result
    return verdict, None
