from __future__ import annotations

import os
from pathlib import Path


def load_runtime_env() -> None:
    """Load local .env values when python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Running without python-dotenv is acceptable.
        pass


def get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_predictions_path(default: str = "artifacts/reports/latest_predictions.csv") -> Path:
    return Path(get_env_str("PREDICTIONS_FILE", default))
