from __future__ import annotations

import numpy as np
import pandas as pd


def simple_directional_backtest(
    returns: pd.Series,
    proba_up: pd.Series,
    threshold: float = 0.55,
    transaction_cost_bps: float = 10.0,
) -> dict[str, float]:
    signal = (proba_up >= threshold).astype(int)
    strategy_ret = signal.shift(1).fillna(0) * returns.fillna(0)
    trades = signal.diff().abs().fillna(0)
    transaction_cost = trades * (transaction_cost_bps / 10000.0)
    net_ret = strategy_ret - transaction_cost

    equity = (1 + net_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    sharpe = 0.0
    if net_ret.std(ddof=1) and net_ret.std(ddof=1) > 0:
        sharpe = float((net_ret.mean() / net_ret.std(ddof=1)) * np.sqrt(252))
    return {
        "net_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "hit_ratio": float((strategy_ret > 0).mean()) if len(strategy_ret) else 0.0,
        "turnover": float(trades.mean()) if len(trades) else 0.0,
        "sharpe": sharpe,
    }
