from __future__ import annotations

import numpy as np
import pandas as pd


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("ticker")
    out["return_1d"] = grouped["Close"].pct_change()
    out["log_return_1d"] = grouped["Close"].transform(lambda s: np.log(s / s.shift(1)))
    return out


def add_rolling_volatility(df: pd.DataFrame, windows: tuple[int, ...] = (10, 20)) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("ticker")
    for window in windows:
        out[f"volatility_{window}"] = grouped["return_1d"].transform(lambda s: s.rolling(window).std())
    return out


def add_return_lags(df: pd.DataFrame, lags: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("ticker")
    for lag in lags:
        out[f"lag_return_{lag}"] = grouped["return_1d"].shift(lag)
    return out


def add_volume_change_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Volume" not in out.columns:
        out["volume_change_1d"] = np.nan
        out["volume_sma_20"] = np.nan
        out["volume_ratio_20"] = np.nan
        return out
    grouped = out.groupby("ticker")
    out["volume_change_1d"] = grouped["Volume"].pct_change()
    out["volume_sma_20"] = grouped["Volume"].transform(lambda s: s.rolling(20).mean())
    out["volume_ratio_20"] = out["Volume"] / out["volume_sma_20"].replace(0, pd.NA)
    return out


def add_return_and_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["ticker", "date"])
    out = add_daily_returns(out)
    out = add_rolling_volatility(out, windows=(10, 20))
    out = add_return_lags(out, lags=(1, 5, 20))
    out = add_volume_change_features(out)
    return out
