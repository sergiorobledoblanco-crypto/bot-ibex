from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import Ridge


def build_regressor(model_name: str = "ridge", random_state: int = 42) -> Any:
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                random_state=random_state,
            )
        except ImportError:
            pass
    return Ridge(alpha=1.0, random_state=random_state)


def train_regressor(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "ridge") -> Any:
    model = build_regressor(model_name=model_name)
    model.fit(X_train, y_train)
    return model
