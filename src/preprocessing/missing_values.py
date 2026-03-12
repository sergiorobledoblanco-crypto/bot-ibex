from __future__ import annotations

import pandas as pd


def fill_missing_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ohlc_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out[ohlc_cols] = out.groupby("ticker")[ohlc_cols].ffill()
    out[ohlc_cols] = out.groupby("ticker")[ohlc_cols].bfill()
    return out
