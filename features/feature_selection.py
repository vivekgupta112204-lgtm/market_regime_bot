"""Automatic feature selection for the HMM pipeline.

Applies two sequential filters:

1. **Constant-column removal** — drops features with zero variance.
2. **High-correlation pruning** — iteratively drops one member of every
   pair whose absolute Pearson correlation exceeds a configurable
   threshold (default 0.95).

The module is deliberately simple; more sophisticated methods (mutual
information, recursive elimination) can be added in later phases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def remove_constant_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """Drop features with zero variance (constant columns).

    Args:
        df: DataFrame containing the features.
        feature_columns: Candidate feature column names.

    Returns:
        Filtered list of feature names with constant columns removed.
    """
    kept: list[str] = []
    dropped: list[str] = []

    for col in feature_columns:
        if df[col].nunique() <= 1:
            dropped.append(col)
        else:
            kept.append(col)

    if dropped:
        logger.info(
            "Removed {} constant column(s): {}",
            len(dropped),
            dropped,
        )
    else:
        logger.info("No constant columns detected")

    return kept


def remove_correlated_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """Remove highly correlated features.

    For every pair with ``|r| > threshold`` the feature that appears
    later in the candidate list is dropped.  This simple heuristic
    keeps the first-listed (typically more fundamental) feature.

    Args:
        df: DataFrame containing the features.
        feature_columns: Candidate feature column names (order matters).
        threshold: Absolute correlation cut-off.

    Returns:
        Filtered list of feature names.
    """
    if len(feature_columns) < 2:
        return list(feature_columns)

    corr_matrix = df[feature_columns].corr().abs()
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)

    to_drop: set[str] = set()
    for col in upper_triangle.columns:
        correlated = upper_triangle.index[upper_triangle[col] > threshold].tolist()
        to_drop.update(correlated)

    kept = [c for c in feature_columns if c not in to_drop]

    if to_drop:
        logger.info(
            "Removed {} correlated feature(s) (threshold={:.2f}): {}",
            len(to_drop),
            threshold,
            sorted(to_drop),
        )
    else:
        logger.info(
            "No features exceed correlation threshold {:.2f}", threshold
        )

    return kept


def select_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    correlation_threshold: float = 0.95,
) -> list[str]:
    """Run the full feature-selection pipeline.

    Applies constant-column removal followed by correlation pruning.

    Args:
        df: DataFrame with computed features.
        feature_columns: Initial candidate features.
        correlation_threshold: Absolute correlation cut-off.

    Returns:
        Final list of selected feature names.
    """
    logger.info(
        "Feature selection: starting with {} candidates",
        len(feature_columns),
    )

    # Step 1 — drop constant columns
    filtered = remove_constant_columns(df, feature_columns)

    # Step 2 — drop highly correlated pairs
    filtered = remove_correlated_features(df, filtered, threshold=correlation_threshold)

    logger.info(
        "Feature selection complete: {} → {} features retained",
        len(feature_columns),
        len(filtered),
    )

    # Print the selected feature set
    for i, feat in enumerate(filtered, 1):
        logger.info("  [{:>2}] {}", i, feat)

    return filtered
