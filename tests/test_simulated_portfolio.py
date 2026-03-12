from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.simulated_portfolio import monte_carlo_gbm_portfolio, monte_carlo_terminal_values


def test_monte_carlo_terminal_values_shape() -> None:
    positions = pd.DataFrame(
        {
            "current_value": [6000.0, 4000.0],
            "retorno_esperado": [0.001, 0.0005],
            "riesgo": [0.02, 0.03],
        }
    )
    out = monte_carlo_terminal_values(positions, horizon_days=30, n_sims=1000)
    assert len(out) == 1000
    assert np.isfinite(out).all()


def test_monte_carlo_gbm_portfolio_outputs() -> None:
    dates = pd.date_range("2025-01-01", periods=300, freq="B")
    history = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "Close": np.concatenate(
                [
                    np.linspace(100, 120, len(dates)),
                    np.linspace(50, 62, len(dates)),
                ]
            ),
        }
    )
    positions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "current_value": [7000.0, 3000.0],
            "retorno_esperado": [0.001, 0.0007],
            "riesgo": [0.02, 0.03],
        }
    )
    out = monte_carlo_gbm_portfolio(
        history_df=history,
        positions_df=positions,
        horizon_days=60,
        n_sims=1000,
        lookback_days=252,
    )
    assert out["paths"].shape[0] == 1000
    assert out["paths"].shape[1] == 61
    assert len(out["terminal_values"]) == 1000
    assert {"mean", "p05", "p50", "p95"} <= set(out["stats"].keys())
