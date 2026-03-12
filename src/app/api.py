from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from runtime_config import get_predictions_path, load_runtime_env


app = FastAPI(title="IBEX 35 Predictor API")
load_runtime_env()
PREDICTIONS_PATH = get_predictions_path()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predictions/latest")
def latest_predictions(limit: int = 20) -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="Predictions file not found.")
    df = pd.read_csv(PREDICTIONS_PATH)
    return df.tail(limit).to_dict(orient="records")
