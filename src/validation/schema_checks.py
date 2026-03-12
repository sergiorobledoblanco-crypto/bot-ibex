from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["date", "ticker", "Open", "High", "Low", "Close", "Volume"]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("'date' column must be datetime")
