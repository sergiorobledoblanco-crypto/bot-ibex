from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, model_dir: str | Path, model_name: str, metadata: dict[str, Any]) -> Path:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / f"{model_name}.joblib"
    meta_path = path / f"{model_name}.metadata.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def load_model(model_path: str | Path) -> Any:
    return joblib.load(model_path)
