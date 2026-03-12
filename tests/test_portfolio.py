from __future__ import annotations

import pandas as pd

from analytics.portfolio import PortfolioConfig, build_daily_portfolio, build_performance_summary


def test_build_daily_portfolio_and_summary() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "empresa": ["A", "B", "A", "B"],
            "prob_subida": [0.7, 0.6, 0.72, 0.58],
            "retorno_esperado": [0.01, 0.008, 0.011, 0.007],
            "riesgo": [0.02, 0.03, 0.02, 0.03],
            "ratio_retorno_riesgo": [0.50, 0.26, 0.55, 0.23],
            "target_return": [0.01, -0.005, 0.008, 0.002],
        }
    )
    holdings, perf = build_daily_portfolio(
        df,
        PortfolioConfig(top_n=2, min_prob=0.55, transaction_cost_bps=0.0),
    )
    assert not holdings.empty
    assert not perf.empty
    assert "equity_curve" in perf.columns
    summary = build_performance_summary(perf)
    assert "net_return" in summary
