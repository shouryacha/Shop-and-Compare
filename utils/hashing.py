"""Checksum helpers used to fingerprint downloaded image bytes."""

from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()
