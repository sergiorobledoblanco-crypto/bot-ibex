from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yfinance as yf


PORTFOLIO_DIR = Path("artifacts/portfolio")
STATE_FILE = PORTFOLIO_DIR / "portfolio_state.json"
OPERATIONS_FILE = PORTFOLIO_DIR / "operations.csv"
HISTORY_FILE = PORTFOLIO_DIR / "portfolio_value_history.csv"
OPEN_POSITIONS_FILE = PORTFOLIO_DIR / "open_positions.csv"
CLOSED_TRADES_FILE = PORTFOLIO_DIR / "closed_trades.csv"


@dataclass(frozen=True)
class PortfolioState:
    initial_capital: float
    cash: float


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    # Supports mixed ISO formats generated across app versions.
    return pd.to_datetime(series, errors="coerce", format="mixed", utc=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_storage() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    if not OPERATIONS_FILE.exists():
        pd.DataFrame(
            columns=[
                "timestamp",
                "side",
                "ticker",
                "empresa",
                "shares",
                "price",
                "amount",
                "realized_pnl",
            ]
        ).to_csv(OPERATIONS_FILE, index=False)
    if not OPEN_POSITIONS_FILE.exists():
        pd.DataFrame(
            columns=[
                "position_id",
                "ticker",
                "empresa",
                "buy_timestamp",
                "buy_price",
                "invested_amount",
                "shares",
            ]
        ).to_csv(OPEN_POSITIONS_FILE, index=False)
    if not CLOSED_TRADES_FILE.exists():
        pd.DataFrame(
            columns=[
                "position_id",
                "ticker",
                "empresa",
                "buy_timestamp",
                "sell_timestamp",
                "buy_price",
                "sell_price",
                "invested_amount",
                "shares",
                "realized_pnl",
                "realized_return",
            ]
        ).to_csv(CLOSED_TRADES_FILE, index=False)
    if not HISTORY_FILE.exists():
        pd.DataFrame(columns=["timestamp", "cash", "positions_value", "total_value"]).to_csv(
            HISTORY_FILE, index=False
        )
    if not STATE_FILE.exists():
        pd.DataFrame([{"initial_capital": 10000.0, "cash": 10000.0}]).to_json(
            STATE_FILE, orient="records", indent=2
        )


def load_portfolio_state() -> PortfolioState:
    _ensure_storage()
    rec = pd.read_json(STATE_FILE).iloc[0]
    return PortfolioState(initial_capital=float(rec["initial_capital"]), cash=float(rec["cash"]))


def save_portfolio_state(state: PortfolioState) -> None:
    _ensure_storage()
    pd.DataFrame([{"initial_capital": state.initial_capital, "cash": state.cash}]).to_json(
        STATE_FILE, orient="records", indent=2
    )


def load_operations() -> pd.DataFrame:
    _ensure_storage()
    ops = pd.read_csv(OPERATIONS_FILE)
    # Backward compatibility with older operation files.
    for col, default in {
        "timestamp": "",
        "side": "",
        "ticker": "",
        "empresa": "",
        "shares": 0.0,
        "price": 0.0,
        "amount": 0.0,
        "realized_pnl": 0.0,
    }.items():
        if col not in ops.columns:
            ops[col] = default
    if "empresa" in ops.columns:
        ops["empresa"] = ops["empresa"].fillna("")
    if not ops.empty:
        ops["timestamp"] = _parse_mixed_datetime(ops["timestamp"])
    return ops


def load_open_positions() -> pd.DataFrame:
    _ensure_storage()
    df = pd.read_csv(OPEN_POSITIONS_FILE)
    if df.empty:
        return df
    if "buy_timestamp" in df.columns:
        df["buy_timestamp"] = _parse_mixed_datetime(df["buy_timestamp"])
    return df


def save_open_positions(df: pd.DataFrame) -> None:
    _ensure_storage()
    df.to_csv(OPEN_POSITIONS_FILE, index=False)


def load_closed_trades() -> pd.DataFrame:
    _ensure_storage()
    df = pd.read_csv(CLOSED_TRADES_FILE)
    if df.empty:
        return df
    if "buy_timestamp" in df.columns:
        df["buy_timestamp"] = _parse_mixed_datetime(df["buy_timestamp"])
    if "sell_timestamp" in df.columns:
        df["sell_timestamp"] = _parse_mixed_datetime(df["sell_timestamp"])
    return df


def save_closed_trades(df: pd.DataFrame) -> None:
    _ensure_storage()
    df.to_csv(CLOSED_TRADES_FILE, index=False)


def save_operations(operations: pd.DataFrame) -> None:
    _ensure_storage()
    operations.to_csv(OPERATIONS_FILE, index=False)


def reset_portfolio(initial_capital: float = 10000.0) -> None:
    _ensure_storage()
    save_portfolio_state(PortfolioState(initial_capital=float(initial_capital), cash=float(initial_capital)))
    pd.DataFrame(
        columns=["timestamp", "side", "ticker", "empresa", "shares", "price", "amount", "realized_pnl"]
    ).to_csv(OPERATIONS_FILE, index=False)
    pd.DataFrame(columns=["timestamp", "cash", "positions_value", "total_value"]).to_csv(
        HISTORY_FILE, index=False
    )
    pd.DataFrame(
        columns=[
            "position_id",
            "ticker",
            "empresa",
            "buy_timestamp",
            "buy_price",
            "invested_amount",
            "shares",
        ]
    ).to_csv(OPEN_POSITIONS_FILE, index=False)
    pd.DataFrame(
        columns=[
            "position_id",
            "ticker",
            "empresa",
            "buy_timestamp",
            "sell_timestamp",
            "buy_price",
            "sell_price",
            "invested_amount",
            "shares",
            "realized_pnl",
            "realized_return",
        ]
    ).to_csv(CLOSED_TRADES_FILE, index=False)


def fetch_latest_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    data = yf.download(
        tickers=tickers,
        period="1d",
        interval="1m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )
    prices: dict[str, float] = {}
    for t in tickers:
        try:
            series = data[t]["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
            prices[t] = float(series.dropna().iloc[-1])
        except Exception:
            pass
    return prices


def _compute_positions_from_operations(ops: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if ops.empty:
        return pd.DataFrame(columns=["ticker", "shares", "avg_cost", "invested"]), 0.0

    positions: dict[str, dict[str, float]] = {}
    realized_total = 0.0
    ops_sorted = ops.sort_values("timestamp")
    for _, row in ops_sorted.iterrows():
        ticker = str(row["ticker"])
        side = str(row["side"]).upper()
        shares = float(row["shares"])
        price = float(row["price"])
        positions.setdefault(ticker, {"shares": 0.0, "avg_cost": 0.0})
        pos = positions[ticker]

        if side == "BUY":
            new_shares = pos["shares"] + shares
            if new_shares > 0:
                pos["avg_cost"] = (pos["avg_cost"] * pos["shares"] + price * shares) / new_shares
            pos["shares"] = new_shares
        elif side == "SELL":
            shares_to_sell = min(shares, pos["shares"])
            realized_total += (price - pos["avg_cost"]) * shares_to_sell
            pos["shares"] -= shares_to_sell
            if pos["shares"] <= 1e-10:
                pos["shares"] = 0.0
                pos["avg_cost"] = 0.0

    pos_rows = []
    for ticker, p in positions.items():
        if p["shares"] > 0:
            pos_rows.append(
                {
                    "ticker": ticker,
                    "shares": p["shares"],
                    "avg_cost": p["avg_cost"],
                    "invested": p["shares"] * p["avg_cost"],
                }
            )
    return pd.DataFrame(pos_rows), realized_total


def build_open_positions_with_market(
    operations: pd.DataFrame,
    prices: dict[str, float],
    company_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    # New lot-based mode: each purchase is an independent position.
    if "position_id" in operations.columns and "buy_price" in operations.columns:
        pos_df = operations.copy()
        if pos_df.empty:
            return pos_df
        company_map = company_map or {}
        pos_df["empresa"] = pos_df.get("empresa", "").replace("", pd.NA).fillna(
            pos_df["ticker"].map(company_map).fillna(pos_df["ticker"])
        )
        pos_df["avg_cost"] = pd.to_numeric(pos_df["buy_price"], errors="coerce").fillna(0.0)
        pos_df["invested"] = pd.to_numeric(pos_df["invested_amount"], errors="coerce").fillna(0.0)
        pos_df["shares"] = pd.to_numeric(pos_df["shares"], errors="coerce").fillna(0.0)
    else:
        pos_df, _ = _compute_positions_from_operations(operations)
    if pos_df.empty:
        return pos_df
    company_map = company_map or {}
    if "empresa" not in pos_df.columns:
        pos_df["empresa"] = pos_df["ticker"].map(company_map).fillna(pos_df["ticker"])
    pos_df["current_price"] = pos_df["ticker"].map(prices).fillna(pos_df["avg_cost"])
    pos_df["current_value"] = pos_df["shares"] * pos_df["current_price"]
    pos_df["unrealized_pnl"] = pos_df["current_value"] - pos_df["invested"]
    pos_df["unrealized_return"] = np.where(
        pos_df["invested"] > 0, pos_df["unrealized_pnl"] / pos_df["invested"], 0.0
    )
    return pos_df.sort_values("current_value", ascending=False).reset_index(drop=True)


def execute_buy(
    ticker: str,
    amount_eur: float,
    market_price: float,
    company_name: str | None = None,
) -> dict[str, Any]:
    state = load_portfolio_state()
    operations = load_operations()
    open_positions = load_open_positions()

    amount = float(max(amount_eur, 0.0))
    price = float(max(market_price, 1e-9))
    if amount <= 0:
        return {"ok": False, "message": "El monto de compra debe ser mayor que 0."}
    if amount > state.cash:
        return {"ok": False, "message": "No hay efectivo suficiente para esta compra."}

    shares = amount / price
    position_id = str(uuid4())
    open_row = {
        "position_id": position_id,
        "ticker": ticker,
        "empresa": company_name or ticker,
        "buy_timestamp": _utc_now(),
        "buy_price": price,
        "invested_amount": amount,
        "shares": shares,
    }
    open_positions = pd.concat([open_positions, pd.DataFrame([open_row])], ignore_index=True)
    save_open_positions(open_positions)

    new_row = {
        "timestamp": _utc_now(),
        "side": "BUY",
        "ticker": ticker,
        "empresa": company_name or ticker,
        "shares": shares,
        "price": price,
        "amount": amount,
        "realized_pnl": 0.0,
    }
    operations = pd.concat([operations, pd.DataFrame([new_row])], ignore_index=True)
    save_operations(operations)
    save_portfolio_state(PortfolioState(initial_capital=state.initial_capital, cash=state.cash - amount))
    return {
        "ok": True,
        "message": f"Compra registrada: {ticker}, {shares:.4f} acciones.",
        "position_id": position_id,
    }


def execute_sell_position(
    position_id: str,
    market_price: float,
    shares_to_sell: float | None = None,
) -> dict[str, Any]:
    state = load_portfolio_state()
    operations = load_operations()
    open_positions = load_open_positions()
    closed_trades = load_closed_trades()

    pos = open_positions[open_positions["position_id"] == position_id]
    if pos.empty:
        return {"ok": False, "message": "No existe la posicion seleccionada."}
    row = pos.iloc[0]
    shares_total = float(row["shares"])
    buy_price = float(row["buy_price"])
    invested = float(row["invested_amount"])
    ticker = str(row["ticker"])
    empresa = str(row.get("empresa", ticker))
    sell_shares = shares_total if shares_to_sell is None else float(shares_to_sell)
    if sell_shares <= 0:
        return {"ok": False, "message": "Las acciones a vender deben ser mayores que 0."}
    if sell_shares > shares_total:
        return {"ok": False, "message": "No puedes vender mas acciones de las disponibles en esta posicion."}

    sell_price = float(max(market_price, 1e-9))
    amount = sell_shares * sell_price
    invested_sold = invested * (sell_shares / shares_total) if shares_total > 0 else 0.0
    realized_pnl = amount - invested_sold
    realized_return = (realized_pnl / invested_sold) if invested_sold > 0 else 0.0

    closed_row = {
        "position_id": position_id,
        "ticker": ticker,
        "empresa": empresa,
        "buy_timestamp": row["buy_timestamp"],
        "sell_timestamp": _utc_now(),
        "buy_price": buy_price,
        "sell_price": sell_price,
        "invested_amount": invested_sold,
        "shares": sell_shares,
        "realized_pnl": realized_pnl,
        "realized_return": realized_return,
    }
    closed_trades = pd.concat([closed_trades, pd.DataFrame([closed_row])], ignore_index=True)
    save_closed_trades(closed_trades)
    remaining_shares = shares_total - sell_shares
    if remaining_shares <= 1e-10:
        open_positions = open_positions[open_positions["position_id"] != position_id].reset_index(drop=True)
    else:
        remaining_invested = invested - invested_sold
        open_positions.loc[open_positions["position_id"] == position_id, "shares"] = remaining_shares
        open_positions.loc[open_positions["position_id"] == position_id, "invested_amount"] = remaining_invested
    save_open_positions(open_positions)

    op_row = {
        "timestamp": _utc_now(),
        "side": "SELL",
        "ticker": ticker,
        "empresa": empresa,
        "shares": sell_shares,
        "price": sell_price,
        "amount": amount,
        "realized_pnl": realized_pnl,
    }
    operations = pd.concat([operations, pd.DataFrame([op_row])], ignore_index=True)
    save_operations(operations)
    save_portfolio_state(PortfolioState(initial_capital=state.initial_capital, cash=state.cash + amount))
    if remaining_shares <= 1e-10:
        msg = f"Posicion {position_id} vendida completamente."
    else:
        msg = f"Venta parcial registrada en {position_id}. Quedan {remaining_shares:.4f} acciones."
    return {"ok": True, "message": msg, "realized_pnl": realized_pnl}


def execute_sell(
    ticker: str,
    shares_to_sell: float,
    market_price: float,
    company_name: str | None = None,
) -> dict[str, Any]:
    state = load_portfolio_state()
    operations = load_operations()
    positions, _ = _compute_positions_from_operations(operations)
    pos_row = positions[positions["ticker"] == ticker]
    if pos_row.empty:
        return {"ok": False, "message": "No hay posicion abierta para este ticker."}

    available_shares = float(pos_row.iloc[0]["shares"])
    shares = float(max(shares_to_sell, 0.0))
    if shares <= 0:
        return {"ok": False, "message": "Las acciones a vender deben ser mayores que 0."}
    if shares > available_shares:
        return {"ok": False, "message": "No puedes vender mas acciones de las disponibles."}

    price = float(max(market_price, 1e-9))
    amount = shares * price
    avg_cost = float(pos_row.iloc[0]["avg_cost"])
    realized_pnl = (price - avg_cost) * shares
    new_row = {
        "timestamp": _utc_now(),
        "side": "SELL",
        "ticker": ticker,
        "empresa": company_name or ticker,
        "shares": shares,
        "price": price,
        "amount": amount,
        "realized_pnl": realized_pnl,
    }
    operations = pd.concat([operations, pd.DataFrame([new_row])], ignore_index=True)
    save_operations(operations)
    save_portfolio_state(PortfolioState(initial_capital=state.initial_capital, cash=state.cash + amount))
    return {"ok": True, "message": f"Venta registrada: {ticker}, {shares:.4f} acciones."}


def append_portfolio_snapshot(
    cash: float,
    positions_value: float,
    min_interval_seconds: int = 15,
    epsilon: float = 1e-6,
) -> None:
    _ensure_storage()
    history = pd.read_csv(HISTORY_FILE)
    now = _utc_now()
    total_value = float(cash + positions_value)
    if not history.empty:
        history["timestamp"] = _parse_mixed_datetime(history["timestamp"])
        last = history.iloc[-1]
        last_ts = pd.to_datetime(last["timestamp"], errors="coerce", utc=True)
        if pd.notna(last_ts):
            elapsed = (pd.to_datetime(now, utc=True) - last_ts).total_seconds()
            same_total = abs(float(last["total_value"]) - total_value) <= epsilon
            if elapsed < min_interval_seconds and same_total:
                return
    row = {
        "timestamp": now,
        "cash": float(cash),
        "positions_value": float(positions_value),
        "total_value": total_value,
    }
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)


def load_portfolio_history() -> pd.DataFrame:
    _ensure_storage()
    hist = pd.read_csv(HISTORY_FILE)
    if not hist.empty:
        hist["timestamp"] = _parse_mixed_datetime(hist["timestamp"])
    return hist


def monte_carlo_terminal_values(
    positions_df: pd.DataFrame,
    horizon_days: int = 126,
    n_sims: int = 2000,
) -> np.ndarray:
    if positions_df.empty:
        return np.array([])

    current_values = positions_df["current_value"].to_numpy(dtype="float64")
    total_now = float(current_values.sum())
    if total_now <= 0:
        return np.array([])
    weights = current_values / total_now
    mus = pd.to_numeric(positions_df.get("retorno_esperado", 0.0), errors="coerce").fillna(0.0).to_numpy()
    sigmas = pd.to_numeric(positions_df.get("riesgo", 0.02), errors="coerce").fillna(0.02).to_numpy()
    sigmas = np.clip(sigmas, 1e-6, 1.0)

    z = np.random.normal(size=(n_sims, len(weights), horizon_days))
    daily = mus[None, :, None] + sigmas[None, :, None] * z
    daily = np.clip(daily, -0.2, 0.2)
    portfolio_daily = np.tensordot(daily, weights, axes=([1], [0]))
    growth = np.prod(1.0 + portfolio_daily, axis=1)
    return total_now * growth


def build_log_returns_matrix(history_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if history_df.empty or not tickers:
        return pd.DataFrame()
    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    prices = (
        df[df["ticker"].isin(tickers)]
        .pivot(index="date", columns="ticker", values="Close")
        .sort_index()
    )
    prices = prices.ffill().dropna(how="all")
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.dropna(how="all")


def estimate_gbm_parameters(
    history_df: pd.DataFrame,
    tickers: list[str],
    lookback_days: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_returns = build_log_returns_matrix(history_df, tickers=tickers)
    if log_returns.empty:
        n = len(tickers)
        return np.zeros(n), np.full(n, 0.02), np.eye(n)

    if len(log_returns) > lookback_days:
        log_returns = log_returns.iloc[-lookback_days:]

    # Keep columns aligned to requested tickers
    for t in tickers:
        if t not in log_returns.columns:
            log_returns[t] = 0.0
    log_returns = log_returns[tickers].fillna(0.0)

    mu = log_returns.mean().to_numpy(dtype="float64")
    sigma = log_returns.std(ddof=1).fillna(0.0).to_numpy(dtype="float64")
    sigma = np.clip(sigma, 1e-6, 1.0)

    corr = log_returns.corr().fillna(0.0).to_numpy(dtype="float64")
    np.fill_diagonal(corr, 1.0)
    # Numerical safety for Cholesky
    corr = corr + np.eye(corr.shape[0]) * 1e-8
    return mu, sigma, corr


def simulate_gbm_portfolio_paths(
    current_values: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    horizon_days: int = 126,
    n_sims: int = 2000,
) -> np.ndarray:
    """
    Simulate correlated GBM paths for a multi-asset portfolio.
    Returns array with shape (n_sims, horizon_days + 1) of portfolio values.
    """
    if current_values.size == 0:
        return np.empty((0, 0))

    n_assets = current_values.shape[0]
    dt = 1.0
    start_value = float(current_values.sum())
    if start_value <= 0:
        return np.empty((0, 0))
    weights = current_values / start_value

    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(corr + np.eye(n_assets) * 1e-6)

    z = np.random.normal(size=(n_sims, horizon_days, n_assets))
    corr_z = np.einsum("sth,ha->sta", z, chol)

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion_scale = sigma * np.sqrt(dt)
    log_increments = drift[None, None, :] + diffusion_scale[None, None, :] * corr_z
    log_relatives = np.cumsum(log_increments, axis=1)
    asset_relatives = np.exp(log_relatives)

    weighted_rel = np.einsum("sta,a->st", asset_relatives, weights)
    paths = np.empty((n_sims, horizon_days + 1), dtype="float64")
    paths[:, 0] = start_value
    paths[:, 1:] = start_value * weighted_rel
    return paths


def monte_carlo_gbm_portfolio(
    history_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    horizon_days: int = 126,
    n_sims: int = 2000,
    lookback_days: int = 252,
) -> dict[str, Any]:
    if positions_df.empty:
        return {"paths": np.empty((0, 0)), "terminal_values": np.array([]), "stats": {}}

    work = positions_df.copy()
    work["current_value"] = pd.to_numeric(work["current_value"], errors="coerce").fillna(0.0)
    work = work[work["current_value"] > 0].copy()
    if work.empty:
        return {"paths": np.empty((0, 0)), "terminal_values": np.array([]), "stats": {}}

    tickers = work["ticker"].astype(str).tolist()
    current_values = work["current_value"].to_numpy(dtype="float64")
    mu, sigma, corr = estimate_gbm_parameters(history_df, tickers=tickers, lookback_days=lookback_days)
    paths = simulate_gbm_portfolio_paths(
        current_values=current_values,
        mu=mu,
        sigma=sigma,
        corr=corr,
        horizon_days=horizon_days,
        n_sims=n_sims,
    )
    if paths.size == 0:
        return {"paths": np.empty((0, 0)), "terminal_values": np.array([]), "stats": {}}
    terminal = paths[:, -1]
    stats = {
        "mean": float(np.mean(terminal)),
        "p05": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
    }
    return {"paths": paths, "terminal_values": terminal, "stats": stats}
