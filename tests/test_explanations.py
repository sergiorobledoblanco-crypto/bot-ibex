from __future__ import annotations

from explanations.signal_rules import generate_signal_explanation


def test_generate_signal_explanation_positive_case() -> None:
    row = {
        "accion": "Comprar",
        "rsi_14": 61.0,
        "macd": 0.4,
        "momentum_10": 0.03,
        "sma_5": 105.0,
        "sma_20": 100.0,
        "sma_50": 95.0,
        "volatility_20": 0.018,
        "volume_change_1d": 0.12,
    }
    out = generate_signal_explanation(row)
    assert out["explanation_title"].startswith("Senal positiva")
    assert len(out["explanation_points"]) >= 6
    assert out["rule_score"] > 0
