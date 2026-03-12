from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_returns(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    if "return_1d" not in out.columns:
        return out
    low, high = out["return_1d"].quantile([lower_q, upper_q]).to_list()
    out["return_1d"] = np.clip(out["return_1d"], low, high)
    return out
