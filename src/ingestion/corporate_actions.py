from __future__ import annotations

import pandas as pd


def apply_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """Use Adj Close as Close when available."""
    out = df.copy()
    if "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]
    return out
