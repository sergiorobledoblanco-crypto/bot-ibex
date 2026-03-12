from __future__ import annotations

import pandas as pd

from evaluation.metrics import regression_metrics
from modeling.train_regression import train_regressor


def fit_and_evaluate_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2,
    model_name: str = "ridge",
) -> dict[str, float]:
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target_return"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target_return"]

    model = train_regressor(X_train, y_train, model_name=model_name)
    y_pred = model.predict(X_test)
    return regression_metrics(y_test, y_pred)
