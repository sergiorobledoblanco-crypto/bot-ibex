from __future__ import annotations

from pathlib import Path

import pandas as pd

from serving.predict import predict_with_model


def run_batch_inference(
    model_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    feature_cols: list[str],
) -> Path:
    df = pd.read_parquet(features_path)
    pred = predict_with_model(model_path=model_path, features_df=df, feature_cols=feature_cols)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    return out_path
