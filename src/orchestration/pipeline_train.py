from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from evaluation.metrics import classification_metrics
from evaluation.reports import write_markdown_report
from features.market_regime import add_market_regime
from features.returns_volatility import add_return_and_volatility_features
from features.target_builder import add_targets
from features.technical import add_technical_features
from free_mode import enforce_free_mode
from ingestion.corporate_actions import apply_adjusted_close
from ingestion.download_history import download_and_store_raw, load_raw_prices
from modeling.model_registry import save_model
from modeling.train_classification import train_classifier
from preprocessing.align_calendar import align_business_calendar
from preprocessing.clean_prices import clean_prices
from preprocessing.missing_values import fill_missing_ohlc
from runtime_config import get_env_str, load_runtime_env
from validation.data_quality import assert_quality, quality_report
from validation.schema_checks import validate_schema


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_training_pipeline(config_path: str | Path = "config/settings.yaml") -> dict[str, float]:
    cfg = load_yaml(config_path)
    enforce_free_mode(cfg)
    tickers = load_yaml("config/tickers_ibex35.yaml")["tickers"]

    raw_file = download_and_store_raw(
        tickers=tickers,
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        raw_dir=cfg["data"]["raw_path"],
        interval=cfg["data"]["interval"],
    )
    df = load_raw_prices(raw_file)
    validate_schema(df)
    assert_quality(quality_report(df))

    df = apply_adjusted_close(df)
    df = clean_prices(df)
    df = align_business_calendar(df)
    df = fill_missing_ohlc(df)

    df = add_technical_features(df)
    df = add_return_and_volatility_features(df)
    df = add_market_regime(df)
    df = add_targets(df, horizon_days=cfg["modeling"]["horizon_days"])
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {"date", "ticker", "target_direction", "target_return", "Adj Close"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    split_idx = int(len(df) * (1 - cfg["modeling"]["test_size"]))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target_direction"].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target_direction"].astype(int)

    model = train_classifier(
        X_train,
        y_train,
        model_name=cfg["modeling"]["classification_model"],
    )
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = classification_metrics(y_test, y_pred, y_proba=y_proba)

    save_model(
        model=model,
        model_dir="artifacts/models",
        model_name="classification_model",
        metadata={"feature_cols": feature_cols, "metrics": metrics},
    )

    Path(cfg["data"]["feature_path"]).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(cfg["data"]["feature_path"]) / "feature_table.parquet", index=False)
    write_markdown_report("artifacts/reports/train_report.md", {"classification": metrics})
    return metrics


if __name__ == "__main__":
    load_runtime_env()
    results = run_training_pipeline(config_path=get_env_str("CONFIG_PATH", "config/settings.yaml"))
    print(results)
