"""Unit tests for Phase 2 — feature engineering, scaling, selection, and HMM training.

All tests run in-memory with synthetic data — no network calls, API keys,
or disk persistence required.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from config.constants import OHLCV_COLUMNS, ScalerType
from features.feature_engineering import (
    add_atr,
    add_bollinger_bands,
    add_candle_features,
    add_daily_return,
    add_ema,
    add_log_return,
    add_macd,
    add_obv,
    add_rsi,
    add_rolling_volatility,
    add_sma,
    add_stochastic,
    compute_all_features,
)
from features.feature_selection import (
    remove_constant_columns,
    remove_correlated_features,
    select_features,
)
from features.scaler import FeatureScaler
from models.hmm_trainer import HMMTrainer, _compute_aic, _compute_bic, _count_parameters
from models.hmm_selector import format_comparison_table, select_best_model
from models.regime_mapper import map_regimes


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

def _make_ohlcv(rows: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data suitable for feature engineering."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    close = 100.0 + rng.standard_normal(rows).cumsum()
    close = np.abs(close) + 50.0  # Ensure positive
    high = close + rng.uniform(0.5, 3.0, rows)
    low = close - rng.uniform(0.5, 3.0, rows)
    low = np.maximum(low, 1.0)  # Ensure positive
    open_ = low + rng.uniform(0, 1, rows) * (high - low)
    volume = rng.integers(10_000, 1_000_000, rows).astype(float)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Feature Engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Tests for individual feature functions and the orchestrator."""

    def test_daily_return(self) -> None:
        df = _make_ohlcv(50)
        result = add_daily_return(df)
        assert "daily_return" in result.columns
        assert result["daily_return"].iloc[0] != result["daily_return"].iloc[0]  # NaN

    def test_log_return(self) -> None:
        df = _make_ohlcv(50)
        result = add_log_return(df)
        assert "log_return" in result.columns

    def test_rolling_volatility(self) -> None:
        df = _make_ohlcv(50)
        df = add_log_return(df)
        result = add_rolling_volatility(df, window=10)
        assert "rolling_volatility" in result.columns

    def test_atr(self) -> None:
        df = _make_ohlcv(50)
        result = add_atr(df, period=14)
        assert "atr" in result.columns

    def test_bollinger_bands(self) -> None:
        df = _make_ohlcv(50)
        result = add_bollinger_bands(df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns

    def test_ema(self) -> None:
        df = _make_ohlcv(100)
        result = add_ema(df)
        assert "ema_20" in result.columns
        assert "ema_50" in result.columns
        assert "ema_distance" in result.columns

    def test_sma(self) -> None:
        df = _make_ohlcv(100)
        result = add_sma(df)
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns

    def test_macd(self) -> None:
        df = _make_ohlcv(50)
        result = add_macd(df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_rsi(self) -> None:
        df = _make_ohlcv(50)
        result = add_rsi(df, period=14)
        assert "rsi" in result.columns

    def test_stochastic(self) -> None:
        df = _make_ohlcv(50)
        result = add_stochastic(df)
        assert "stochastic_k" in result.columns

    def test_obv(self) -> None:
        df = _make_ohlcv(50)
        result = add_obv(df)
        assert "obv" in result.columns

    def test_candle_features(self) -> None:
        df = _make_ohlcv(50)
        result = add_candle_features(df)
        assert "high_low_range" in result.columns
        assert "candle_body" in result.columns
        assert "upper_wick" in result.columns
        assert "lower_wick" in result.columns

    def test_compute_all_features(self) -> None:
        df = _make_ohlcv(200)
        result = compute_all_features(df)
        # Should have added many columns.
        assert len(result.columns) > len(OHLCV_COLUMNS)
        # Should have no NaN values.
        assert result.isna().sum().sum() == 0
        # Should have fewer rows (warm-up dropped).
        assert len(result) < 200


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------

class TestScaler:
    """Tests for the FeatureScaler class."""

    def test_standard_scaler(self) -> None:
        df = _make_ohlcv(100)
        df = compute_all_features(df)
        cols = ["daily_return", "log_return"]
        scaler = FeatureScaler(ScalerType.STANDARD)
        scaled = scaler.fit_transform(df, cols)
        assert scaler.is_fitted
        # Scaled columns should have ~0 mean.
        assert abs(scaled["daily_return"].mean()) < 0.1

    def test_robust_scaler(self) -> None:
        df = _make_ohlcv(100)
        df = compute_all_features(df)
        cols = ["daily_return"]
        scaler = FeatureScaler(ScalerType.ROBUST)
        scaled = scaler.fit_transform(df, cols)
        assert scaler.is_fitted

    def test_minmax_scaler(self) -> None:
        df = _make_ohlcv(100)
        df = compute_all_features(df)
        cols = ["daily_return"]
        scaler = FeatureScaler(ScalerType.MINMAX)
        scaled = scaler.fit_transform(df, cols)
        assert scaled["daily_return"].min() >= -0.01  # Numerical tolerance
        assert scaled["daily_return"].max() <= 1.01

    def test_save_and_load(self) -> None:
        df = _make_ohlcv(100)
        df = compute_all_features(df)
        cols = ["daily_return", "log_return"]
        scaler = FeatureScaler(ScalerType.ROBUST)
        scaler.fit(df, cols)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "scaler.pkl"
            scaler.save(path)
            loaded = FeatureScaler.load(path)
            assert loaded.is_fitted
            assert loaded.feature_names == cols

    def test_unfitted_transform_raises(self) -> None:
        df = _make_ohlcv(50)
        df = compute_all_features(df)
        scaler = FeatureScaler(ScalerType.STANDARD)
        with pytest.raises(RuntimeError):
            scaler.transform(df)


# ---------------------------------------------------------------------------
# Feature Selection tests
# ---------------------------------------------------------------------------

class TestFeatureSelection:
    """Tests for feature selection utilities."""

    def test_remove_constant_columns(self) -> None:
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [5, 5, 5],  # Constant
            "c": [7, 8, 9],
        })
        kept = remove_constant_columns(df, ["a", "b", "c"])
        assert "b" not in kept
        assert "a" in kept
        assert "c" in kept

    def test_remove_correlated(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        df = pd.DataFrame({
            "a": x,
            "b": x + rng.normal(0, 0.001, 100),  # Nearly identical to a
            "c": rng.standard_normal(100),
        })
        kept = remove_correlated_features(df, ["a", "b", "c"], threshold=0.95)
        # b should be dropped.
        assert "b" not in kept
        assert "a" in kept
        assert "c" in kept

    def test_select_features_full_pipeline(self) -> None:
        df = _make_ohlcv(200)
        df = compute_all_features(df)
        feature_cols = [c for c in df.columns if c not in OHLCV_COLUMNS]
        selected = select_features(df, feature_cols, correlation_threshold=0.95)
        assert len(selected) > 0
        assert len(selected) <= len(feature_cols)


# ---------------------------------------------------------------------------
# HMM Trainer tests
# ---------------------------------------------------------------------------

class TestHMMTrainer:
    """Tests for HMM training and model selection."""

    def test_parameter_counting(self) -> None:
        n_params = _count_parameters(n_states=3, n_features=5, covariance_type="full")
        # 3*(3-1) + (3-1) + 3*5 + 3*(5*6/2) = 6 + 2 + 15 + 45 = 68
        assert n_params == 68

    def test_aic_bic(self) -> None:
        aic = _compute_aic(-100.0, 10)
        bic = _compute_bic(-100.0, 10, 1000)
        assert aic == 220.0
        assert bic > 200.0

    def test_train_single(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        trainer = HMMTrainer(n_iter=50, n_init=2, random_state=42)
        result = trainer.train_single(X, n_states=2)
        assert result.model is not None
        assert result.log_likelihood > -np.inf
        assert result.n_states == 2

    def test_train_range(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        trainer = HMMTrainer(n_iter=50, n_init=2, random_state=42)
        results = trainer.train_range(X, min_states=2, max_states=3)
        assert len(results) == 2

    def test_select_best_model(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        trainer = HMMTrainer(n_iter=50, n_init=2, random_state=42)
        results = trainer.train_range(X, min_states=2, max_states=4)
        selection = select_best_model(results, criterion="bic")
        assert selection.best_result.model is not None

    def test_comparison_table(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        trainer = HMMTrainer(n_iter=50, n_init=2, random_state=42)
        results = trainer.train_range(X, min_states=2, max_states=3)
        table = format_comparison_table(results)
        assert "States" in table
        assert "AIC" in table


# ---------------------------------------------------------------------------
# Regime Mapper tests
# ---------------------------------------------------------------------------

class TestRegimeMapper:
    """Tests for regime labelling."""

    def test_map_regimes(self) -> None:
        df = _make_ohlcv(200)
        df = compute_all_features(df)
        n = len(df)

        rng = np.random.default_rng(42)
        hidden_states = rng.integers(0, 3, size=n)

        stats, label_map = map_regimes(df, hidden_states, n_states=3)
        assert len(stats) == 3
        assert len(label_map) == 3
        # Should have Bull and Bear at minimum.
        labels = set(label_map.values())
        assert "Bull Market" in labels
        assert "Bear Market" in labels

    def test_two_state_mapping(self) -> None:
        df = _make_ohlcv(100)
        df = compute_all_features(df)
        n = len(df)

        hidden_states = np.zeros(n, dtype=int)
        hidden_states[n // 2:] = 1

        stats, label_map = map_regimes(df, hidden_states, n_states=2)
        assert len(label_map) == 2
