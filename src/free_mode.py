from __future__ import annotations

from pathlib import Path

from runtime_config import get_env_bool


def _is_local_host(host: str) -> bool:
    normalized = host.strip().lower()
    return normalized in {"127.0.0.1", "localhost"}


def enforce_free_mode(cfg: dict) -> None:
    """
    Validate configuration for strictly no-cost operation.
    """
    free_mode = get_env_bool("FREE_MODE", True)
    if not free_mode:
        return

    provider = cfg.get("data", {}).get("provider", "")
    if str(provider).lower() != "yfinance":
        raise ValueError("FREE_MODE requires 'data.provider: yfinance'.")

    for key in ("raw_path", "processed_path", "feature_path", "dataset_path"):
        path_value = cfg.get("data", {}).get(key, "")
        if not str(path_value).startswith("data/"):
            raise ValueError(f"FREE_MODE requires local data path under 'data/': {key}")

    predictions_file = cfg.get("serving", {}).get("predictions_file", "")
    if not str(predictions_file).startswith("artifacts/"):
        raise ValueError("FREE_MODE requires predictions_file under 'artifacts/'.")

    host = str(cfg.get("serving", {}).get("host", "127.0.0.1"))
    if not _is_local_host(host):
        raise ValueError("FREE_MODE requires serving.host to be localhost/127.0.0.1")

    # Ensure destination dirs exist locally.
    for relative_path in [
        cfg["data"]["raw_path"],
        cfg["data"]["processed_path"],
        cfg["data"]["feature_path"],
        cfg["data"]["dataset_path"],
        str(Path(predictions_file).parent),
    ]:
        Path(relative_path).mkdir(parents=True, exist_ok=True)
