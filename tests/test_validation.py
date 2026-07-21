"""Unit tests for the validation engine and helper utilities.

These tests run purely in-memory with synthetic DataFrames — no network
calls or API keys required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from config.constants import OHLCV_COLUMNS
from utils.helpers import (
    build_filename,
    is_cache_valid,
    ms_to_utc_datetime,
    save_dataframe,
    to_utc_timestamp,
)
from utils.validation import (
    ValidationReport,
    clean_ohlcv,
    validate_ohlcv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(
    rows: int = 100,
    start: str = "2023-01-01",
    freq: str = "D",
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame.

    Args:
        rows: Number of rows.
        start: Start date.
        freq: Pandas frequency string.

    Returns:
        Well-formed DataFrame that should pass validation.
    """
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(start=start, periods=rows, freq=freq, tz="UTC")
    close = 100.0 + rng.standard_normal(rows).cumsum()
    high = close + rng.uniform(0.5, 2.0, rows)
    low = close - rng.uniform(0.5, 2.0, rows)
    open_ = low + rng.uniform(0, 1, rows) * (high - low)
    volume = rng.integers(1_000, 1_000_000, rows).astype(float)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for ``utils.validation``."""

    def test_valid_data_passes(self) -> None:
        """Well-formed data should produce a passing report."""
        df = _make_ohlcv()
        report = validate_ohlcv(df, timeframe="1d")
        assert report.passed is True
        assert report.duplicate_count == 0
        assert len(report.invalid_ohlc_indices) == 0

    def test_duplicate_detection(self) -> None:
        """Duplicated timestamps should be detected."""
        df = _make_ohlcv(rows=50)
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        report = validate_ohlcv(df, timeframe="1d", max_duplicate_pct=0.0)
        assert report.duplicate_count > 0
        assert report.passed is False

    def test_invalid_ohlc_detection(self) -> None:
        """Rows with high < low should be flagged."""
        df = _make_ohlcv(rows=20)
        df.loc[5, "high"] = df.loc[5, "low"] - 1.0
        report = validate_ohlcv(df, timeframe="1d")
        assert 5 in report.invalid_ohlc_indices

    def test_missing_volume_detection(self) -> None:
        """Zero / NaN volume should be flagged."""
        df = _make_ohlcv(rows=20)
        df.loc[3, "volume"] = 0.0
        df.loc[7, "volume"] = np.nan
        report = validate_ohlcv(df, timeframe="1d")
        assert 3 in report.missing_volume_indices
        assert 7 in report.missing_volume_indices

    def test_ordering_violation(self) -> None:
        """Out-of-order timestamps should be caught."""
        df = _make_ohlcv(rows=20)
        # Swap rows 5 and 10 to create a time reversal.
        df.iloc[5], df.iloc[10] = df.iloc[10].copy(), df.iloc[5].copy()
        report = validate_ohlcv(df, timeframe="1d")
        assert len(report.ordering_violations) > 0

    def test_clean_ohlcv_removes_duplicates(self) -> None:
        """``clean_ohlcv`` should deduplicate and sort."""
        df = _make_ohlcv(rows=30)
        dirty = pd.concat([df, df.iloc[:5]], ignore_index=True)
        cleaned = clean_ohlcv(dirty)
        assert len(cleaned) == 30
        # Verify sorted.
        diffs = pd.to_datetime(cleaned["timestamp"]).diff().dropna()
        assert (diffs >= pd.Timedelta(0)).all()

    def test_report_summary_string(self) -> None:
        """The summary should be a non-empty string."""
        df = _make_ohlcv(rows=10)
        report = validate_ohlcv(df, timeframe="1d")
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for ``utils.helpers``."""

    def test_build_filename(self) -> None:
        """Filename should encode all parameters."""
        name = build_filename("SPY", "1d", "2020-01-01", "2025-01-01", "yahoo")
        assert name == "yahoo_SPY_1d_20200101_20250101"

    def test_to_utc_timestamp_naive(self) -> None:
        """Naïve datetimes should be assumed UTC."""
        naive = datetime(2023, 6, 15, 12, 0, 0)
        result = to_utc_timestamp(naive)
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_ms_to_utc_datetime(self) -> None:
        """Epoch millis should convert to correct UTC datetime."""
        epoch_ms = 1_700_000_000_000  # 2023-11-14T22:13:20Z
        result = ms_to_utc_datetime(epoch_ms)
        assert result.year == 2023
        assert result.tzinfo == timezone.utc

    def test_save_and_load(self) -> None:
        """Round-trip save / load should preserve row count."""
        df = _make_ohlcv(rows=20)
        with TemporaryDirectory() as tmp:
            paths = save_dataframe(df, Path(tmp), "test_data")
            assert "csv" in paths
            assert "parquet" in paths
            loaded = pd.read_parquet(paths["parquet"])
            assert len(loaded) == 20

    def test_cache_validity_missing_file(self) -> None:
        """Non-existent cache file should return False."""
        assert is_cache_valid(Path("/nonexistent/file.parquet"), 24) is False
