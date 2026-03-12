from __future__ import annotations

import pandas as pd

from analytics.expected_return import compute_expected_return


def test_expected_return_columns_created() -> None:
    df = pd.DataFrame(
        {
            "proba_up": [0.62, 0.41],
            "return_1d": [0.01, -0.015],
            "sma_5": [102, 98],
            "sma_20": [100, 100],
            "sma_50": [99, 101],
            "volatility_20": [0.018, 0.045],
            "rsi_14": [58, 74],
            "macd": [0.4, -0.2],
            "macd_signal": [0.2, -0.1],
            "momentum_10": [0.03, -0.04],
            "volume_change_1d": [0.12, -0.08],
        }
    )
    out = compute_expected_return(df)
    assert "retorno_esperado" in out.columns
    assert "riesgo" in out.columns
    assert "ratio_retorno_riesgo" in out.columns
    assert out["riesgo"].min() > 0
