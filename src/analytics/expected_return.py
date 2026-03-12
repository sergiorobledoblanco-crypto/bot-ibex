from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpectedReturnWeights:
    alpha_model: float = 0.30
    short_term_return: float = 0.10
    signal_strength: float = 0.08
    trend: float = 0.17
    momentum: float = 0.12
    macd: float = 0.08
    rsi: float = 0.06
    volatility_penalty: float = 0.22


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _clip(series: pd.Series, min_v: float, max_v: float) -> pd.Series:
    return series.clip(lower=min_v, upper=max_v)


def compute_alpha_model(df: pd.DataFrame) -> pd.Series:
    prob = _safe_series(df, "proba_up", default=0.5)
    return _clip(prob - 0.5, -0.5, 0.5) / 0.5


def compute_short_term_return_component(df: pd.DataFrame) -> pd.Series:
    ret_1d = _safe_series(df, "return_1d", default=0.0)
    return np.tanh(ret_1d / 0.02)


def compute_signal_strength_component(df: pd.DataFrame) -> pd.Series:
    prob = _safe_series(df, "proba_up", default=0.5)
    direction = np.sign(prob - 0.5)
    strength = (prob - 0.5).abs().mul(2.0)
    return _clip(direction * strength, -1.0, 1.0)


def compute_trend_component(df: pd.DataFrame) -> pd.Series:
    sma5 = _safe_series(df, "sma_5")
    sma20 = _safe_series(df, "sma_20")
    sma50 = _safe_series(df, "sma_50")
    short_trend = np.where(sma5 > sma20, 1.0, -1.0)
    medium_trend = np.where(sma20 > sma50, 1.0, -1.0)
    return pd.Series((short_trend + medium_trend) / 2.0, index=df.index)


def compute_momentum_component(df: pd.DataFrame) -> pd.Series:
    momentum = _safe_series(df, "momentum_10", default=0.0)
    return np.tanh(momentum / 0.05)


def compute_macd_component(df: pd.DataFrame) -> pd.Series:
    macd = _safe_series(df, "macd")
    signal = _safe_series(df, "macd_signal")
    diff = macd - signal
    return np.tanh(diff / 0.5)


def compute_rsi_component(df: pd.DataFrame) -> pd.Series:
    rsi = _safe_series(df, "rsi_14", default=50.0)
    out = pd.Series(0.0, index=df.index, dtype="float64")
    out = np.where(rsi > 70, -_clip((rsi - 70) / 20, 0.0, 1.0), out)
    out = np.where((rsi >= 50) & (rsi <= 65), _clip((rsi - 50) / 15, 0.0, 1.0) * 0.6, out)
    out = np.where(rsi < 35, -_clip((35 - rsi) / 20, 0.0, 1.0) * 0.4, out)
    return pd.Series(out, index=df.index)


def compute_volatility_penalty(df: pd.DataFrame) -> pd.Series:
    vol = _safe_series(df, "volatility_20", default=0.02)
    return np.tanh(_clip(vol, 0.0, 0.20) / 0.03)


def compute_expected_return(df: pd.DataFrame, weights: ExpectedReturnWeights | None = None) -> pd.DataFrame:
    """
    Build interpretable expected return from normalized components.
    """
    w = weights or ExpectedReturnWeights()
    out = df.copy()

    out["alpha_model_component"] = compute_alpha_model(out)
    out["short_term_component"] = compute_short_term_return_component(out)
    out["signal_strength_component"] = compute_signal_strength_component(out)
    out["trend_component"] = compute_trend_component(out)
    out["momentum_component"] = compute_momentum_component(out)
    out["macd_component"] = compute_macd_component(out)
    out["rsi_component"] = compute_rsi_component(out)
    out["volatility_penalty_component"] = compute_volatility_penalty(out)

    out["expected_return_score"] = (
        w.alpha_model * out["alpha_model_component"]
        + w.short_term_return * out["short_term_component"]
        + w.signal_strength * out["signal_strength_component"]
        + w.trend * out["trend_component"]
        + w.momentum * out["momentum_component"]
        + w.macd * out["macd_component"]
        + w.rsi * out["rsi_component"]
        - w.volatility_penalty * out["volatility_penalty_component"]
    )

    base_move = _clip(0.005 + 0.5 * _safe_series(out, "return_1d", default=0.0).abs(), 0.005, 0.03)
    out["retorno_esperado"] = np.tanh(out["expected_return_score"]) * base_move

    out["riesgo"] = _clip(_safe_series(out, "volatility_20", default=0.02), 1e-6, 1.0)
    out["ratio_retorno_riesgo"] = out["retorno_esperado"] / out["riesgo"]
    return out


def build_expected_return_reason(row: pd.Series) -> str:
    positive_reasons: list[str] = []
    negative_reasons: list[str] = []

    if float(row.get("alpha_model_component", 0.0)) > 0.2:
        positive_reasons.append("probabilidad elevada")
    if float(row.get("trend_component", 0.0)) > 0:
        positive_reasons.append("tendencia positiva")
    if float(row.get("momentum_component", 0.0)) > 0:
        positive_reasons.append("momentum favorable")
    if float(row.get("macd_component", 0.0)) > 0:
        positive_reasons.append("confirmacion MACD")

    if float(row.get("volatility_penalty_component", 0.0)) > 0.55:
        negative_reasons.append("volatilidad alta")
    if float(row.get("rsi_component", 0.0)) < -0.15:
        negative_reasons.append("RSI de sobrecompra")
    if float(row.get("trend_component", 0.0)) < 0:
        negative_reasons.append("tendencia debil")

    if positive_reasons and not negative_reasons:
        return "Retorno esperado alto por " + ", ".join(positive_reasons) + "."
    if negative_reasons and not positive_reasons:
        return "Retorno esperado penalizado por " + ", ".join(negative_reasons) + "."
    if positive_reasons and negative_reasons:
        return (
            "Retorno esperado mixto: fortalezas en "
            + ", ".join(positive_reasons)
            + " pero penalizado por "
            + ", ".join(negative_reasons)
            + "."
        )
    return "Retorno esperado moderado sin una señal dominante clara."
