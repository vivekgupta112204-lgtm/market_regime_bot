"""Configurable feature scaler with save / load support.

Wraps scikit-learn scalers behind a unified interface and persists the
fitted scaler to disk so that the same transform can be applied at
inference time without re-fitting.

Supported scaling strategies:

* **StandardScaler** — zero mean, unit variance.
* **RobustScaler** — median / IQR based; resilient to outliers.
* **MinMaxScaler** — scales to [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config.constants import ScalerType


# Type alias for the union of all supported scaler objects.
ScalerInstance = StandardScaler | RobustScaler | MinMaxScaler


class FeatureScaler:
    """Create, fit, transform, and persist a feature scaler.

    Args:
        scaler_type: One of ``ScalerType.STANDARD``, ``ROBUST``, or
            ``MINMAX``.
    """

    def __init__(self, scaler_type: ScalerType = ScalerType.ROBUST) -> None:
        self.scaler_type = scaler_type
        self._scaler: ScalerInstance = self._build_scaler(scaler_type)
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

    @staticmethod
    def _build_scaler(scaler_type: ScalerType) -> ScalerInstance:
        """Instantiate the underlying scikit-learn scaler.

        Args:
            scaler_type: Enum member identifying the strategy.

        Returns:
            An un-fitted scaler instance.

        Raises:
            ValueError: When the scaler type is not recognised.
        """
        if scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        if scaler_type == ScalerType.ROBUST:
            return RobustScaler()
        if scaler_type == ScalerType.MINMAX:
            return MinMaxScaler()
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> "FeatureScaler":
        """Fit the scaler on the given feature columns.

        Args:
            df: DataFrame containing the feature columns.
            feature_columns: Column names to scale.

        Returns:
            ``self``, for method chaining.
        """
        self._feature_names = list(feature_columns)
        self._scaler.fit(df[feature_columns].values)
        self._is_fitted = True
        logger.info(
            "Scaler fitted ({}) on {} features",
            self.scaler_type.value,
            len(feature_columns),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler and return a new DataFrame.

        Args:
            df: DataFrame containing the feature columns used during
                fitting.

        Returns:
            A copy of *df* with scaled feature columns.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        scaled = self._scaler.transform(df[self._feature_names].values)
        result = df.copy()
        result[self._feature_names] = scaled
        logger.debug("Scaler transform applied to {} columns", len(self._feature_names))
        return result

    def fit_transform(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """Fit the scaler and return the transformed DataFrame.

        Args:
            df: DataFrame containing the feature columns.
            feature_columns: Column names to scale.

        Returns:
            Scaled DataFrame.
        """
        self.fit(df, feature_columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling transformation.

        Args:
            df: Scaled DataFrame.

        Returns:
            DataFrame with original-scale feature values.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        original = self._scaler.inverse_transform(df[self._feature_names].values)
        result = df.copy()
        result[self._feature_names] = original
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Serialize the fitted scaler to disk.

        Args:
            path: Destination file path (e.g. ``saved_models/scaler.pkl``).

        Returns:
            The path that was written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self._scaler,
            "scaler_type": self.scaler_type.value,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(payload, path)
        logger.info("Scaler saved → {}", path.name)
        return path

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        """Deserialize a previously saved scaler.

        Args:
            path: Path to the serialized file.

        Returns:
            A fully-restored ``FeatureScaler`` instance.

        Raises:
            FileNotFoundError: When *path* does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        payload = joblib.load(path)
        instance = cls(scaler_type=ScalerType(payload["scaler_type"]))
        instance._scaler = payload["scaler"]
        instance._feature_names = payload["feature_names"]
        instance._is_fitted = payload["is_fitted"]
        logger.info("Scaler loaded ← {} ({} features)", path.name, len(instance._feature_names))
        return instance

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature column names the scaler was fitted on."""
        return list(self._feature_names)

    @property
    def is_fitted(self) -> bool:
        """Whether the scaler has been fitted."""
        return self._is_fitted
