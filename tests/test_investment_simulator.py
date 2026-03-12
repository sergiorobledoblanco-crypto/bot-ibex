from __future__ import annotations

import pandas as pd

from analytics.investment_simulator import run_investment_simulation


def test_run_investment_simulation_outputs_metrics() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * 5 + ["BBB"] * 5,
            "Close": [100, 101, 102, 101, 103, 50, 49, 50, 51, 52],
            "return_1d": [0.0, 0.01, 0.01, -0.0098, 0.0198, 0.0, -0.02, 0.0204, 0.02, 0.0196],
        }
    )
    out = run_investment_simulation(
        history_df=df,
        selected_tickers=["AAA", "BBB"],
        initial_capital=10000,
        raw_weights={"AAA": 0.6, "BBB": 0.4},
    )
    assert out["portfolio_metrics"].final_value > 0
    assert "benchmark_ibex_value" in out
    assert "benchmark_equal_value" in out
