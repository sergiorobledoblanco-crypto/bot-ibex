from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from analytics.expected_return import build_expected_return_reason, compute_expected_return
from explanations.signal_rules import append_signal_explanations
from features.feature_explanations import DEFAULT_DISPLAY_FEATURES
from free_mode import enforce_free_mode
from runtime_config import get_env_str, load_runtime_env
from serving.batch_inference import run_batch_inference


def _load_company_names(path: str | Path = "config/company_names.yaml") -> dict[str, str]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    return data.get("companies", {})


def _build_action_label(proba_up: float | None, threshold: float) -> str:
    if proba_up is None:
        return "Sin señal"
    if proba_up >= threshold:
        return "Comprar"
    return "No comprar"


def run_prediction_pipeline(config_path: str | Path = "config/settings.yaml") -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    enforce_free_mode(cfg)
    model_meta = json.loads(
        Path("artifacts/models/classification_model.metadata.json").read_text(encoding="utf-8")
    )
    feature_cols = model_meta["feature_cols"]
    feature_path = Path(cfg["data"]["feature_path"]) / "feature_table.parquet"
    output_path = Path(cfg["serving"]["predictions_file"])

    run_batch_inference(
        model_path="artifacts/models/classification_model.joblib",
        features_path=feature_path,
        output_path=output_path,
        feature_cols=feature_cols,
    )

    pred_all = pd.read_csv(output_path)
    features_df = pd.read_parquet(feature_path)
    pred_all["date"] = pd.to_datetime(pred_all["date"])
    features_df["date"] = pd.to_datetime(features_df["date"])

    selected_cols = (
        ["date", "ticker", "target_return"]
        + [c for c in DEFAULT_DISPLAY_FEATURES if c in features_df.columns]
    )
    universe = pred_all.merge(features_df[selected_cols], on=["date", "ticker"], how="left")
    universe["empresa"] = universe["ticker"].map(_load_company_names())
    threshold = float(cfg.get("evaluation", {}).get("probability_threshold", 0.55))
    universe["accion"] = universe["proba_up"].apply(lambda p: _build_action_label(p, threshold))
    universe["confianza"] = universe["proba_up"]
    universe["prob_subida"] = universe["proba_up"]
    universe = compute_expected_return(universe)
    universe["explicacion_retorno"] = universe.apply(build_expected_return_reason, axis=1)

    pred = universe.sort_values(["date", "ticker"]).groupby("ticker").tail(1)
    pred = append_signal_explanations(pred)
    pred = pred.sort_values("ratio_retorno_riesgo", ascending=False).reset_index(drop=True)

    ordered_cols = [
        "date",
        "empresa",
        "ticker",
        "accion",
        "confianza",
        "prob_subida",
        "retorno_esperado",
        "riesgo",
        "ratio_retorno_riesgo",
        "explicacion_retorno",
        "explanation_title",
        "explanation_points",
        "rule_score",
        "prediction",
    ] + [c for c in DEFAULT_DISPLAY_FEATURES if c in pred.columns]
    pred = pred[ordered_cols]
    pred["date"] = pred["date"].dt.strftime("%Y-%m-%d")
    pred.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    load_runtime_env()
    path = run_prediction_pipeline(config_path=get_env_str("CONFIG_PATH", "config/settings.yaml"))
    print(f"Predictions written to: {path}")
