from __future__ import annotations

import pandas as pd

from modeling.train_classification import train_classifier


def test_train_classifier_runs() -> None:
    X = pd.DataFrame({"f1": [0.1, 0.2, -0.1, -0.3, 0.5, 0.7], "f2": [1, 0, 1, 0, 1, 0]})
    y = pd.Series([1, 1, 0, 0, 1, 1])
    model = train_classifier(X, y, model_name="logistic")
    preds = model.predict(X)
    assert len(preds) == len(X)
