"""
Centralized configuration for Shop & Compare.

Design intent:
    - This is the ONLY place that knows about environment variables and
      directory layout. Every other module receives what it needs via
      function/constructor arguments — nothing else should call
      `os.environ` or read `.env` directly.
    - Secrets (API keys) are declared here as optional fields so the shape of
      required configuration is visible in one place, but no key is ever
      hardcoded or given a real default.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment variables / a `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PIR_", extra="ignore")

    # --- Directory layout ---------------------------------------------------
    # Single root: every product's images live at `images_dir/<product_slug>/`.
    # There is no separate scratch/downloads directory - candidate images that
    # haven't been accepted yet are downloaded to an ephemeral OS temp
    # directory (created and cleaned up per-run) instead of anywhere under
    # this tree.
    images_dir: Path = Field(default=Path(__file__).parent / "images")

    # --- Pipeline tuning ---------------------------------------------------
    request_timeout_seconds: float = Field(default=15.0, gt=0)

    # --- Secrets (real values come from the environment / .env, never the code) --
    serpapi_api_key: str | None = None
    # `validation_alias` bypasses `env_prefix` for this one field: the NVIDIA
    # Build key is conventionally set as a bare `NVIDIA_API_KEY` (matching
    # `classifier/nemotron.py` and NVIDIA's own docs), not `PIR_NVIDIA_API_KEY`.
    nvidia_api_key: str | None = Field(default=None, validation_alias="NVIDIA_API_KEY")


def get_settings() -> Settings:
    """
    Return a fresh `Settings` instance.

    A plain function (not a module-level singleton) so tests can construct
    isolated settings without mutating global state.
    """
    return Settings()
