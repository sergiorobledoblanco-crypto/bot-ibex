from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(columns: pd.Index) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flat: list[str] = []
        for col in columns:
            if isinstance(col, tuple):
                flat.append(str(col[0]))
            else:
                flat.append(str(col))
        return flat
    return [str(col) for col in columns]


@dataclass
class YahooFinanceProvider:
    interval: str = "1d"

    def download_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval=self.interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = _flatten_columns(df.columns)
        df = df.rename(columns={"Date": "date"})
        missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {ticker}: {missing}")

        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df.sort_values("date").reset_index(drop=True)

    def download_many(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        frames = [self.download_history(t, start_date, end_date) for t in tickers]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
