"""Model persistence — save and load HMM artefact bundles.

A saved model bundle is a directory containing:

* ``hmm_model.pkl`` — the fitted ``GaussianHMM`` object.
* ``scaler.pkl`` — the fitted ``FeatureScaler``.
* ``feature_list.json`` — ordered list of feature column names.
* ``config.json`` — training configuration snapshot.
* ``metrics.json`` — training metrics (AIC, BIC, log-likelihood, …).
* ``regime_map.json`` — state → label mapping.

The ``ModelLoader`` class provides symmetric ``save`` / ``load`` methods.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger

from models.hmm_trainer import HMMResult
from models.regime_mapper import RegimeStats


class ModelLoader:
    """Save and load HMM model bundles to / from disk.

    A *bundle* is a single directory whose name encodes the symbol,
    timeframe, and timestamp, making it easy to track model lineage.
    """

    @staticmethod
    def save_bundle(
        directory: Path,
        result: HMMResult,
        feature_names: list[str],
        regime_stats: list[RegimeStats],
        label_map: dict[int, str],
        config_snapshot: dict[str, Any],
        scaler_path: Path | None = None,
    ) -> Path:
        """Persist a complete model bundle.

        Args:
            directory: Target directory (created if absent).
            result: The winning ``HMMResult``.
            feature_names: Ordered feature column names.
            regime_stats: Per-state regime statistics.
            label_map: State-ID → label mapping.
            config_snapshot: Dictionary snapshot of training settings.
            scaler_path: If already saved, the path to the scaler file
                (it will be copied into the bundle).

        Returns:
            The bundle directory path.
        """
        directory.mkdir(parents=True, exist_ok=True)

        # 1. Model
        model_path = directory / "hmm_model.pkl"
        joblib.dump(result.model, model_path)
        logger.info("Model saved → {}", model_path.name)

        # 2. Feature list
        feat_path = directory / "feature_list.json"
        feat_path.write_text(
            json.dumps(feature_names, indent=2), encoding="utf-8"
        )
        logger.info("Feature list saved → {}", feat_path.name)

        # 3. Metrics
        metrics = {
            "n_states": result.n_states,
            "log_likelihood": float(result.log_likelihood),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "n_params": result.n_params,
            "train_time_seconds": round(result.train_time_seconds, 3),
            "converged": result.converged,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        metrics_path = directory / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        logger.info("Metrics saved → {}", metrics_path.name)

        # 4. Config snapshot
        config_path = directory / "config.json"
        # Convert non-serializable types.
        safe_config = _make_json_safe(config_snapshot)
        config_path.write_text(
            json.dumps(safe_config, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Config snapshot saved → {}", config_path.name)

        # 5. Regime map
        regime_data = {
            "label_map": {str(k): v for k, v in label_map.items()},
            "state_statistics": [
                {
                    "state_id": rs.state_id,
                    "label": rs.label,
                    "observation_count": rs.observation_count,
                    "avg_return": float(rs.avg_return),
                    "avg_volatility": float(rs.avg_volatility),
                    "avg_close": float(rs.avg_close),
                }
                for rs in regime_stats
            ],
        }
        regime_path = directory / "regime_map.json"
        regime_path.write_text(
            json.dumps(regime_data, indent=2), encoding="utf-8"
        )
        logger.info("Regime map saved → {}", regime_path.name)

        # 6. Copy scaler if provided
        if scaler_path and scaler_path.exists():
            import shutil
            dest = directory / "scaler.pkl"
            shutil.copy2(scaler_path, dest)
            logger.info("Scaler copied → {}", dest.name)

        logger.success("Model bundle saved → {}", directory)
        return directory

    @staticmethod
    def load_bundle(directory: Path) -> dict[str, Any]:
        """Load a previously saved model bundle.

        Args:
            directory: Path to the bundle directory.

        Returns:
            Dictionary with keys:
            ``model``, ``feature_names``, ``metrics``, ``config``,
            ``regime_map``, ``scaler`` (if present).

        Raises:
            FileNotFoundError: If the directory or required files are
                missing.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Model bundle not found: {directory}")

        bundle: dict[str, Any] = {}

        # Model
        model_path = directory / "hmm_model.pkl"
        if model_path.exists():
            bundle["model"] = joblib.load(model_path)
            logger.info("Model loaded ← {}", model_path.name)

        # Feature list
        feat_path = directory / "feature_list.json"
        if feat_path.exists():
            bundle["feature_names"] = json.loads(feat_path.read_text(encoding="utf-8"))

        # Metrics
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            bundle["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))

        # Config
        config_path = directory / "config.json"
        if config_path.exists():
            bundle["config"] = json.loads(config_path.read_text(encoding="utf-8"))

        # Regime map
        regime_path = directory / "regime_map.json"
        if regime_path.exists():
            bundle["regime_map"] = json.loads(regime_path.read_text(encoding="utf-8"))

        # Scaler
        scaler_path = directory / "scaler.pkl"
        if scaler_path.exists():
            bundle["scaler"] = joblib.load(scaler_path)
            logger.info("Scaler loaded ← {}", scaler_path.name)

        logger.success("Model bundle loaded ← {}", directory)
        return bundle


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable types.

    Args:
        obj: Input value.

    Returns:
        A JSON-safe equivalent.
    """
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
