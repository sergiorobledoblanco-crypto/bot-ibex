from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp


def ks_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    drift_scores: dict[str, float] = {}
    for col in feature_cols:
        ref = reference_df[col].dropna()
        cur = current_df[col].dropna()
        if ref.empty or cur.empty:
            drift_scores[col] = 1.0
            continue
        _, p_value = ks_2samp(ref, cur)
        drift_scores[col] = float(p_value)
    return drift_scores
