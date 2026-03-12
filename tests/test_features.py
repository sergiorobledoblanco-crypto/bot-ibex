from __future__ import annotations

import pandas as pd

from features.returns_volatility import add_return_and_volatility_features
from features.technical import add_technical_features
from features.target_builder import add_targets


def test_feature_and_target_generation() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="B"),
            "ticker": ["SAN.MC"] * 30,
            "Close": [10 + i * 0.1 for i in range(30)],
            "Volume": [1_000_000 + i * 5_000 for i in range(30)],
        }
    )
    out = add_technical_features(df)
    out = add_return_and_volatility_features(out)
    out = add_targets(out, horizon_days=1)
    assert "return_1d" in out.columns
    assert "sma_20" in out.columns
    assert "macd" in out.columns
    assert "bb_upper" in out.columns
    assert "momentum_10" in out.columns
    assert "volume_change_1d" in out.columns
    assert "target_direction" in out.columns
