# IBEX 35 Predictor Bot

Proyecto modular para descargar datos del IBEX 35, construir features, entrenar modelos de ML y publicar predicciones en una interfaz sencilla.

## Arquitectura

- `src/ingestion`: descarga de históricos y normalización inicial
- `src/validation`: reglas de calidad y esquema
- `src/preprocessing`: limpieza, calendario y nulos
- `src/features`: ingeniería de variables y objetivos
- `src/modeling`: entrenamiento, CV temporal, tuning y registro
- `src/evaluation`: métricas y backtest simple
- `src/serving`: inferencia batch y monitorización de drift
- `src/app`: API FastAPI y UI Streamlit
- `src/orchestration`: pipelines de entrenamiento y predicción

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración rápida

1. Copia `.env.example` a `.env`
2. Ajusta puertos/rutas si lo necesitas

Variables soportadas:

- `FREE_MODE` (por defecto `true`)
- `CONFIG_PATH` (por defecto `config/settings.yaml`)
- `PREDICTIONS_FILE` (por defecto `artifacts/reports/latest_predictions.csv`)
- `API_HOST`, `API_PORT`
- `UI_HOST`, `UI_PORT`

## Ejecución (un comando)

PowerShell:

```powershell
.\scripts\run_all.ps1
```

## Operación 100% sin coste

Usa este modo para forzar operación gratuita y local:

```powershell
.\scripts\run_free_mode.ps1
```

Opcional (sin reentrenar):

```powershell
.\scripts\run_free_mode.ps1 -SkipTrain
```

Este modo aplica automáticamente:

- `FREE_MODE=true`
- proveedor gratuito `yfinance` únicamente
- host local solo (`127.0.0.1` o `localhost`)
- almacenamiento local en `data/` y `artifacts/`
- telemetría de Streamlit desactivada

Opcional (si ya entrenaste y quieres saltar entrenamiento):

```powershell
.\scripts\run_all.ps1 -SkipTrain
```

Parar API/UI:

```powershell
.\scripts\stop_all.ps1
```

## Ejecución manual por pasos

Antes de ejecutar comandos manuales, define:

```powershell
$env:PYTHONPATH="src"
```

Entrenar modelo y generar tabla de features:

```bash
python -m orchestration.pipeline_train
```

Generar predicciones:

```bash
python -m orchestration.pipeline_predict
```

Levantar API:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Levantar UI:

```bash
python -m streamlit run src/app/ui_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

## Tests

```bash
python -m pytest -q
```

## Variables principales del modelo

- `return_1d`: cambio porcentual diario del cierre.
- `sma_5`, `sma_20`, `sma_50`: medias móviles de 5, 20 y 50 sesiones para tendencia.
- `volatility_20`: desviación estándar rolling de retornos en 20 sesiones (riesgo reciente).
- `rsi_14`: oscilador de fuerza relativa para detectar sobrecompra/sobreventa.
- `macd` y `macd_signal`: impulso de tendencia y su línea de señal.
- `bb_upper` y `bb_lower`: bandas de Bollinger superior/inferior según volatilidad.
- `momentum_10`: fuerza del movimiento comparando con el precio de hace 10 sesiones.
- `volume_change_1d`: variación diaria de volumen negociado.
