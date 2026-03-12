from __future__ import annotations

import pandas as pd


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset=["date", "ticker"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out
