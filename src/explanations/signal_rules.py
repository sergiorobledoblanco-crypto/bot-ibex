from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or pd.isna(value):
        return default
    return float(value)


def generate_signal_explanation(row: Mapping[str, Any]) -> dict[str, Any]:
    bullets: list[str] = []
    positive = 0
    negative = 0

    rsi = _num(row, "rsi_14")
    if rsi > 50:
        bullets.append(f"RSI > 50 ({rsi:.1f}), momentum de mercado positivo.")
        positive += 1
    else:
        bullets.append(f"RSI <= 50 ({rsi:.1f}), impulso limitado o debil.")
        negative += 1

    sma_5 = _num(row, "sma_5")
    sma_20 = _num(row, "sma_20")
    sma_50 = _num(row, "sma_50")
    if sma_5 > sma_20:
        bullets.append("SMA5 > SMA20, tendencia de corto plazo alcista.")
        positive += 1
    else:
        bullets.append("SMA5 <= SMA20, tendencia de corto plazo sin confirmacion alcista.")
        negative += 1

    if sma_20 > sma_50:
        bullets.append("SMA20 > SMA50, estructura de tendencia favorable.")
        positive += 1
    else:
        bullets.append("SMA20 <= SMA50, sesgo de fondo mas debil.")
        negative += 1

    macd = _num(row, "macd")
    if macd > 0:
        bullets.append(f"MACD positivo ({macd:.3f}), sesgo alcista.")
        positive += 1
    else:
        bullets.append(f"MACD no positivo ({macd:.3f}), sesgo menos alcista.")
        negative += 1

    momentum = _num(row, "momentum_10")
    if momentum > 0:
        bullets.append(f"Momentum 10d positivo ({momentum:.2%}), continuidad de movimiento.")
        positive += 1
    else:
        bullets.append(f"Momentum 10d no positivo ({momentum:.2%}), perdida de impulso.")
        negative += 1

    volatility = _num(row, "volatility_20")
    if volatility <= 0.02:
        bullets.append(f"Volatilidad 20d contenida ({volatility:.2%}), entorno mas estable.")
        positive += 1
    elif volatility >= 0.04:
        bullets.append(f"Volatilidad 20d elevada ({volatility:.2%}), riesgo de ruido alto.")
        negative += 1
    else:
        bullets.append(f"Volatilidad 20d media ({volatility:.2%}), riesgo moderado.")

    volume_change = _num(row, "volume_change_1d")
    if volume_change > 0:
        bullets.append(f"Aumento de volumen ({volume_change:.2%}), movimiento con participacion.")
        positive += 1
    else:
        bullets.append(f"Caida de volumen ({volume_change:.2%}), confirmacion mas debil.")
        negative += 1

    action = str(row.get("accion", "No comprar"))
    if action == "Comprar" and positive >= negative:
        title = "Senal positiva porque:"
    elif action == "Comprar":
        title = "Senal de compra con riesgos porque:"
    elif negative > positive:
        title = "Senal negativa porque:"
    else:
        title = "Senal neutral porque:"

    return {
        "explanation_title": title,
        "explanation_points": bullets,
        "rule_score": positive - negative,
    }


def append_signal_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    explanation_data = out.apply(generate_signal_explanation, axis=1)
    out["explanation_title"] = explanation_data.apply(lambda d: d["explanation_title"])
    out["explanation_points"] = explanation_data.apply(lambda d: " || ".join(d["explanation_points"]))
    out["rule_score"] = explanation_data.apply(lambda d: d["rule_score"])
    return out
