from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_classifier(model_name: str = "logistic", random_state: int = 42) -> Any:
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                random_state=random_state,
            )
        except ImportError:
            pass
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "logistic") -> Any:
    model = build_classifier(model_name=model_name)
    model.fit(X_train, y_train)
    return model
