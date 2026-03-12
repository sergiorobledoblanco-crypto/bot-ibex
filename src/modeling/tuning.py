from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from modeling.cv_timeseries import build_time_series_cv


def tune_model(
    model: Any,
    param_distributions: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    n_splits: int = 5,
    scoring: str = "roc_auc",
) -> Any:
    cv = build_time_series_cv(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_
