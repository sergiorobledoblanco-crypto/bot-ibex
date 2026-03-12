from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    top_n: int = 5
    min_prob: float = 0.55
    transaction_cost_bps: float = 10.0


def _inverse_risk_weights(risk: pd.Series) -> pd.Series:
    safe_risk = pd.to_numeric(risk, errors="coerce").fillna(1e-6).clip(lower=1e-6)
    inv = 1.0 / safe_risk
    total = inv.sum()
    if total <= 0:
        return pd.Series(np.zeros(len(inv)), index=inv.index)
    return inv / total


def build_daily_portfolio(
    universe_df: pd.DataFrame,
    config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build top-N daily portfolio and compute realized daily performance.
    Uses target_return as next-day realized return proxy.
    """
    df = universe_df.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].dropna().unique().tolist())

    holdings_rows: list[dict] = []
    perf_rows: list[dict] = []
    prev_weights: dict[str, float] = {}
    equity = 1.0

    for date in dates:
        day = df[df["date"] == date].copy()
        day = day[day["prob_subida"] >= config.min_prob]
        day = day.sort_values("ratio_retorno_riesgo", ascending=False).head(config.top_n)

        if day.empty:
            perf_rows.append(
                {
                    "date": date,
                    "gross_return": 0.0,
                    "turnover": 0.0,
                    "transaction_cost": 0.0,
                    "portfolio_return": 0.0,
                    "equity_curve": equity,
                    "n_positions": 0,
                }
            )
            prev_weights = {}
            continue

        day["weight"] = _inverse_risk_weights(day["riesgo"])
        day["target_return"] = (
            pd.to_numeric(day["target_return"], errors="coerce")
            .fillna(0.0)
            .clip(lower=-0.20, upper=0.20)
        )
        day["weighted_return"] = day["weight"] * day["target_return"]
        gross_return = float(day["weighted_return"].sum())

        curr_weights = dict(zip(day["ticker"], day["weight"]))
        all_tickers = set(prev_weights) | set(curr_weights)
        turnover = float(sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers))
        transaction_cost = turnover * (config.transaction_cost_bps / 10000.0)
        portfolio_return = gross_return - transaction_cost
        equity *= 1.0 + portfolio_return

        for _, row in day.iterrows():
            holdings_rows.append(
                {
                    "date": date,
                    "empresa": row.get("empresa"),
                    "ticker": row["ticker"],
                    "weight": float(row["weight"]),
                    "prob_subida": float(row.get("prob_subida", 0.0)),
                    "retorno_esperado": float(row.get("retorno_esperado", 0.0)),
                    "riesgo": float(row.get("riesgo", 0.0)),
                    "ratio_retorno_riesgo": float(row.get("ratio_retorno_riesgo", 0.0)),
                    "target_return": float(row["target_return"]),
                    "weighted_return": float(row["weighted_return"]),
                }
            )

        perf_rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "portfolio_return": portfolio_return,
                "equity_curve": equity,
                "n_positions": int(len(day)),
            }
        )
        prev_weights = curr_weights

    holdings_df = pd.DataFrame(holdings_rows)
    perf_df = pd.DataFrame(perf_rows).sort_values("date").reset_index(drop=True)
    return holdings_df, perf_df


def build_performance_summary(perf_df: pd.DataFrame) -> dict[str, float]:
    if perf_df.empty:
        return {
            "net_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_positions": 0.0,
        }
    returns = perf_df["portfolio_return"].fillna(0.0)
    equity = perf_df["equity_curve"].ffill().fillna(1.0)
    drawdown = equity / equity.cummax() - 1.0
    sharpe = 0.0
    if returns.std(ddof=1) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252))
    return {
        "net_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "avg_positions": float(perf_df["n_positions"].mean()),
    }


def save_portfolio_outputs(
    holdings_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    base_dir: str | Path = "artifacts/reports",
) -> dict[str, Path]:
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdings_path = out_dir / "portfolio_holdings_daily.csv"
    perf_path = out_dir / "portfolio_performance.csv"
    summary_path = out_dir / "portfolio_summary.csv"
    latest_holdings_path = out_dir / "portfolio_holdings_latest.csv"

    holdings_df.to_csv(holdings_path, index=False)
    perf_df.to_csv(perf_path, index=False)
    pd.DataFrame([build_performance_summary(perf_df)]).to_csv(summary_path, index=False)

    if not holdings_df.empty:
        latest_date = holdings_df["date"].max()
        holdings_df[holdings_df["date"] == latest_date].to_csv(latest_holdings_path, index=False)
    else:
        pd.DataFrame().to_csv(latest_holdings_path, index=False)

    return {
        "holdings_daily": holdings_path,
        "performance": perf_path,
        "summary": summary_path,
        "holdings_latest": latest_holdings_path,
    }
