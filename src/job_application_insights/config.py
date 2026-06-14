"""Application configuration loaded from environment via pydantic-settings.

Demonstrates the Pydantic-v2 settings pattern used throughout the project:
typed, validated config with sensible defaults and a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Top-level application settings.

    Week-0 deliberately uses a plain BaseModel so we don't pin the project to
    pydantic-settings yet — we'll layer that on in Week 1 when API keys arrive.
    """

    data_dir: Path = Field(default=Path("./data"))
    raw_dir: Path = Field(default=Path("./data/raw"))
    processed_dir: Path = Field(default=Path("./data/processed"))


def load_settings() -> Settings:
    """Construct a Settings instance with default values."""
    return Settings()
