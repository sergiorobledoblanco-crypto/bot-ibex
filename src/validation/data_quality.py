from __future__ import annotations

import pandas as pd


def quality_report(df: pd.DataFrame) -> dict[str, float]:
    total_rows = float(len(df))
    duplicates = float(df.duplicated(subset=["date", "ticker"]).sum())
    null_ratio = float(df.isna().sum().sum()) / max(total_rows * max(len(df.columns), 1), 1.0)
    monotonic_ratio = 1.0 if df.sort_values(["ticker", "date"]).equals(df) else 0.0
    return {
        "rows": total_rows,
        "duplicate_rows": duplicates,
        "null_ratio": null_ratio,
        "is_sorted_ratio": monotonic_ratio,
    }


def assert_quality(report: dict[str, float], max_null_ratio: float = 0.01) -> None:
    if report["duplicate_rows"] > 0:
        raise ValueError("Data quality failed: duplicate rows found.")
    if report["null_ratio"] > max_null_ratio:
        raise ValueError(f"Data quality failed: null ratio > {max_null_ratio}")
