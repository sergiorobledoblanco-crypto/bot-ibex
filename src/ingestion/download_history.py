from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ingestion.providers import YahooFinanceProvider


def download_and_store_raw(
    tickers: Iterable[str],
    start_date: str,
    end_date: str | None,
    raw_dir: str | Path,
    interval: str = "1d",
) -> Path:
    provider = YahooFinanceProvider(interval=interval)
    df = provider.download_many(tickers, start_date=start_date, end_date=end_date)
    if df.empty:
        raise ValueError("No data was downloaded from provider.")

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    file_path = raw_path / "ibex_raw.parquet"
    df.to_parquet(file_path, index=False)
    return file_path


def load_raw_prices(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
