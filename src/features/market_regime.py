from __future__ import annotations

import pandas as pd


def add_market_regime(df: pd.DataFrame, benchmark_ticker: str = "IBE.MC") -> pd.DataFrame:
    out = df.copy().sort_values(["ticker", "date"])
    bench = out[out["ticker"] == benchmark_ticker][["date", "Close"]].copy()
    if bench.empty:
        out["market_trend_20"] = 0
        out["market_vol_20"] = 0.0
        return out
    bench["market_return"] = bench["Close"].pct_change()
    bench["market_trend_20"] = (bench["Close"] > bench["Close"].rolling(20).mean()).astype(int)
    bench["market_vol_20"] = bench["market_return"].rolling(20).std()
    out = out.merge(
        bench[["date", "market_trend_20", "market_vol_20"]],
        on="date",
        how="left",
    )
    return out
