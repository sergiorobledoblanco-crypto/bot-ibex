from __future__ import annotations

FEATURE_EXPLANATIONS: dict[str, str] = {
    "return_1d": "Retorno diario del precio de cierre; mide variación de un día a otro.",
    "sma_5": "Media móvil simple de 5 días; tendencia muy corta.",
    "sma_20": "Media móvil simple de 20 días; tendencia de corto/medio plazo.",
    "sma_50": "Media móvil simple de 50 días; tendencia de fondo.",
    "volatility_20": "Volatilidad rolling a 20 días de los retornos; mide riesgo reciente.",
    "rsi_14": "Índice de fuerza relativa a 14 días; sobrecompra/sobreventa.",
    "macd": "Diferencia entre medias exponenciales rápida y lenta; impulso de tendencia.",
    "macd_signal": "Media exponencial del MACD; referencia para cruces de señal.",
    "bb_upper": "Banda superior de Bollinger; zona alta respecto a volatilidad reciente.",
    "bb_lower": "Banda inferior de Bollinger; zona baja respecto a volatilidad reciente.",
    "momentum_10": "Cambio relativo del precio frente a 10 días atrás; fuerza del movimiento.",
    "volume_change_1d": "Cambio diario del volumen negociado; presión compradora/vendedora.",
}

DEFAULT_DISPLAY_FEATURES: list[str] = [
    "return_1d",
    "sma_5",
    "sma_20",
    "sma_50",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "momentum_10",
    "volume_change_1d",
]
