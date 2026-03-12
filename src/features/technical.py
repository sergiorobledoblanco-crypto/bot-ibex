from __future__ import annotations

import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def add_moving_averages(df: pd.DataFrame, periods: tuple[int, ...] = (5, 20, 50)) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("ticker")
    for period in periods:
        out[f"sma_{period}"] = grouped["Close"].transform(lambda s: s.rolling(period).mean())
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("ticker")
    out["ema_fast"] = grouped["Close"].transform(lambda s: s.ewm(span=fast, adjust=False).mean())
    out["ema_slow"] = grouped["Close"].transform(lambda s: s.ewm(span=slow, adjust=False).mean())
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["macd_signal"] = grouped["macd"].transform(lambda s: s.ewm(span=signal, adjust=False).mean())
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    out[f"rsi_{window}"] = out.groupby("ticker")["Close"].transform(lambda s: _rsi(s, window=window))
    return out


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("ticker")
    rolling_mean = grouped["Close"].transform(lambda s: s.rolling(window).mean())
    rolling_std = grouped["Close"].transform(lambda s: s.rolling(window).std())
    out["bb_middle"] = rolling_mean
    out["bb_upper"] = rolling_mean + n_std * rolling_std
    out["bb_lower"] = rolling_mean - n_std * rolling_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"].replace(0, pd.NA)
    return out


def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    out = df.copy()
    out[f"momentum_{period}"] = out.groupby("ticker")["Close"].transform(
        lambda s: s / s.shift(period) - 1.0
    )
    return out


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["ticker", "date"])
    out = add_moving_averages(out, periods=(5, 20, 50))
    out = add_macd(out, fast=12, slow=26, signal=9)
    out = add_rsi(out, window=14)
    out = add_bollinger_bands(out, window=20, n_std=2.0)
    out = add_momentum(out, period=10)
    return out
