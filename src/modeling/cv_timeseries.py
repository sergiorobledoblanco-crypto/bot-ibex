from __future__ import annotations

from sklearn.model_selection import TimeSeriesSplit


def build_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)
