from __future__ import annotations

import pandas as pd

from validation.schema_checks import validate_schema


def test_validate_schema_accepts_minimal_prices() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["SAN.MC", "SAN.MC"],
            "Open": [3.5, 3.6],
            "High": [3.6, 3.7],
            "Low": [3.4, 3.5],
            "Close": [3.55, 3.65],
            "Volume": [1_000_000, 1_200_000],
        }
    )
    validate_schema(df)
