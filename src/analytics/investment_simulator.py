from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    max_drawdown: float


def build_returns_matrix(history_df: pd.DataFrame) -> pd.DataFrame:
    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "return_1d" not in df.columns:
        df = df.sort_values(["ticker", "date"])
        df["return_1d"] = df.groupby("ticker")["Close"].pct_change()
    returns = (
        df.pivot(index="date", columns="ticker", values="return_1d")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return returns


def normalize_weights(selected_tickers: list[str], raw_weights: dict[str, float] | None = None) -> dict[str, float]:
    if not selected_tickers:
        return {}
    if not raw_weights:
        w = 1.0 / len(selected_tickers)
        return {t: w for t in selected_tickers}
    total = float(sum(max(raw_weights.get(t, 0.0), 0.0) for t in selected_tickers))
    if total <= 0:
        w = 1.0 / len(selected_tickers)
        return {t: w for t in selected_tickers}
    return {t: max(raw_weights.get(t, 0.0), 0.0) / total for t in selected_tickers}


def compute_portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    if not weights:
        return pd.Series(dtype="float64")
    selected = [t for t in weights if t in returns.columns]
    if not selected:
        return pd.Series(dtype="float64")
    aligned_weights = np.array([weights[t] for t in selected], dtype="float64")
    matrix = returns[selected].fillna(0.0).to_numpy()
    values = matrix @ aligned_weights
    return pd.Series(values, index=returns.index, name="portfolio_return")


def compute_value_curve(daily_returns: pd.Series, initial_capital: float) -> pd.Series:
    if daily_returns.empty:
        return pd.Series(dtype="float64")
    initial = max(float(initial_capital), 1.0)
    return initial * (1.0 + daily_returns.fillna(0.0)).cumprod()


def compute_performance_metrics(daily_returns: pd.Series, value_curve: pd.Series) -> PerformanceMetrics:
    if daily_returns.empty or value_curve.empty:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    trading_days = max(len(daily_returns), 1)
    total_return = float(value_curve.iloc[-1] / value_curve.iloc[0] - 1.0) if len(value_curve) > 0 else 0.0
    annualized = float((1.0 + total_return) ** (252.0 / trading_days) - 1.0) if total_return > -1.0 else -1.0
    vol = float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    sharpe = float((daily_returns.mean() * 252) / vol) if vol > 0 else 0.0
    drawdown = value_curve / value_curve.cummax() - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    return PerformanceMetrics(
        final_value=float(value_curve.iloc[-1]),
        total_return=total_return,
        annualized_return=annualized,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )


def compute_benchmark_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(index=returns.index)
    out = pd.DataFrame(index=returns.index)
    out["benchmark_equal_ibex"] = returns.mean(axis=1)

    if "^IBEX" in returns.columns:
        out["benchmark_ibex35"] = returns["^IBEX"]
    else:
        large_caps = ["SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "TEF.MC", "REP.MC"]
        present = [t for t in large_caps if t in returns.columns]
        if present:
            out["benchmark_ibex35"] = returns[present].mean(axis=1)
        else:
            out["benchmark_ibex35"] = out["benchmark_equal_ibex"]
    return out


def run_investment_simulation(
    history_df: pd.DataFrame,
    selected_tickers: list[str],
    initial_capital: float,
    raw_weights: dict[str, float] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    returns = build_returns_matrix(history_df)
    if start_date:
        returns = returns[returns.index >= pd.to_datetime(start_date)]
    if end_date:
        returns = returns[returns.index <= pd.to_datetime(end_date)]

    weights = normalize_weights(selected_tickers, raw_weights=raw_weights)
    portfolio_returns = compute_portfolio_returns(returns, weights)
    portfolio_value = compute_value_curve(portfolio_returns, initial_capital=initial_capital)
    portfolio_metrics = compute_performance_metrics(portfolio_returns, portfolio_value)

    benchmarks = compute_benchmark_returns(returns)
    ibex_returns = benchmarks.get("benchmark_ibex35", pd.Series(dtype="float64"))
    equal_returns = benchmarks.get("benchmark_equal_ibex", pd.Series(dtype="float64"))
    ibex_value = compute_value_curve(ibex_returns, initial_capital=initial_capital) if not ibex_returns.empty else pd.Series(dtype="float64")
    equal_value = compute_value_curve(equal_returns, initial_capital=initial_capital) if not equal_returns.empty else pd.Series(dtype="float64")
    ibex_metrics = compute_performance_metrics(ibex_returns, ibex_value) if not ibex_returns.empty else PerformanceMetrics(0, 0, 0, 0, 0, 0)
    equal_metrics = compute_performance_metrics(equal_returns, equal_value) if not equal_returns.empty else PerformanceMetrics(0, 0, 0, 0, 0, 0)

    return {
        "weights": weights,
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "portfolio_metrics": portfolio_metrics,
        "benchmark_ibex_value": ibex_value,
        "benchmark_equal_value": equal_value,
        "benchmark_ibex_metrics": ibex_metrics,
        "benchmark_equal_metrics": equal_metrics,
    }
