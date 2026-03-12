from __future__ import annotations

from pathlib import Path

import pandas as pd

from modeling.model_registry import load_model


def predict_with_model(
    model_path: str | Path,
    features_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    model = load_model(model_path)
    X = features_df[feature_cols].fillna(0)
    out = features_df[["date", "ticker"]].copy()
    if hasattr(model, "predict_proba"):
        out["proba_up"] = model.predict_proba(X)[:, 1]
        out["prediction"] = (out["proba_up"] >= 0.5).astype(int)
    else:
        out["prediction"] = model.predict(X)
    return out
