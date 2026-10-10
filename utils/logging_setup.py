"""Process-wide logging configuration."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger once, at process startup.

    Idempotent-ish in practice: `logging.basicConfig` is a no-op if handlers
    are already configured, so calling this more than once (e.g. in tests)
    is harmless.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
