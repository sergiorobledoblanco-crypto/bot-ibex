from __future__ import annotations

import pandas as pd


def align_business_calendar(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker", as_index=False):
        g = g.sort_values("date")
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="B")
        g = g.set_index("date").reindex(idx)
        g["ticker"] = ticker
        g.index.name = "date"
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)
