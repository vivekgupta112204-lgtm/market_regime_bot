"""End-to-end feature pipeline: load → engineer → scale → select.

This module wires together the individual feature-engineering, scaling,
and selection components into a single callable that Phase 2 training
scripts and Phase 3 signal generators can invoke identically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from config.constants import OHLCV_COLUMNS, ScalerType
from config.settings import Settings
from features.feature_engineering import compute_all_features
from features.feature_selection import select_features
from features.scaler import FeatureScaler


# Columns that are part of the raw OHLCV bar and should NOT be treated
# as model features.
_NON_FEATURE_COLUMNS: set[str] = {
    "timestamp", "open", "high", "low", "close", "volume",
}


def _identify_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of all engineered feature columns.

    Any column that is not in the raw OHLCV set is considered a feature.

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Sorted list of feature column names.
    """
    return sorted(
        col for col in df.columns if col not in _NON_FEATURE_COLUMNS
    )


class FeaturePipeline:
    """Orchestrate the full feature-preparation workflow.

    Args:
        settings: Application settings (provides scaler type,
            correlation threshold, and save paths).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._scaler = FeatureScaler(scaler_type=settings.scaler_type)
        self._feature_columns: list[str] = []
        self._selected_features: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        *,
        scale: bool = True,
        select: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Execute the pipeline: engineer → (scale) → (select).

        Args:
            df: Clean OHLCV DataFrame from Phase 1.
            scale: Whether to apply feature scaling.
            select: Whether to apply feature selection.

        Returns:
            A 2-tuple of:
            - The transformed DataFrame.
            - The final list of selected feature column names.
        """
        logger.info("Feature pipeline starting ({} input rows)", len(df))

        # ---- Step 1: Feature Engineering ----------------------------------
        df = compute_all_features(df)
        self._feature_columns = _identify_feature_columns(df)
        logger.info(
            "Engineered {} features: {}",
            len(self._feature_columns),
            self._feature_columns,
        )

        # ---- Step 2: Feature Selection & Scaling (Pre-PCA) ----------------
        if select:
            self._selected_features = select_features(
                df,
                self._feature_columns,
                correlation_threshold=self._settings.correlation_threshold,
            )
        else:
            self._selected_features = list(self._feature_columns)

        if scale:
            df = self._scaler.fit_transform(df, self._selected_features)

        # ---- Step 3: Dimensionality Reduction (PCA Anti-Overfitting) ------
        # Forces agents to learn underlying orthogonal structures rather than memorizing indicator quirks
        from sklearn.decomposition import PCA
        
        # Determine number of components (max 4, or less if input features are fewer)
        n_components = min(4, len(self._selected_features))
        
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(df[self._selected_features].fillna(0))
        
        pca_cols = [f'pca_{i}' for i in range(n_components)]
        for i, col in enumerate(pca_cols):
            df[col] = pca_data[:, i]
            
        # Drop raw high-dimensional indicators to prevent data leakage/memorization
        df.drop(columns=self._selected_features, inplace=True, errors='ignore')
        
        self._selected_features = pca_cols

        logger.info(
            "Feature pipeline complete: PCA reduced down to {} orthogonal components (evr: {})",
            len(self._selected_features),
            pca.explained_variance_ratio_.round(2)
        )
        # Store PCA object for transform during inference
        self.pca = pca
        return df, self._selected_features

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def scaler(self) -> FeatureScaler:
        """Return the fitted scaler instance."""
        return self._scaler

    @property
    def feature_columns(self) -> list[str]:
        """All engineered feature column names (before selection)."""
        return list(self._feature_columns)

    @property
    def selected_features(self) -> list[str]:
        """Feature column names after selection."""
        return list(self._selected_features)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_scaler(self, directory: Path) -> Path:
        """Persist the fitted scaler and PCA object to the given directory."""
        import joblib
        if hasattr(self, 'pca') and self.pca:
            joblib.dump(self.pca, directory / "pca.pkl")
            
        return self._scaler.save(directory / "scaler.pkl")

    def load_scaler(self, path: Path) -> None:
        """Load a previously saved scaler and accompanying PCA weights."""
        import joblib
        self._scaler = FeatureScaler.load(path)
        
        pca_path = path.parent / "pca.pkl"
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            self._selected_features = [f'pca_{i}' for i in range(self.pca.n_components_)]
        else:
            self._selected_features = self._scaler.feature_names
