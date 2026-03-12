from __future__ import annotations

import pandas as pd


def add_targets(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    out = df.copy().sort_values(["ticker", "date"])
    g = out.groupby("ticker")
    out["target_return"] = g["Close"].shift(-horizon_days) / out["Close"] - 1.0
    out["target_direction"] = (out["target_return"] > 0).astype("Int64")
    return out
