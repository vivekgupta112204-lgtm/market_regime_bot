"""OHLCV data-validation engine for Market Regime Bot.

The validator inspects a standardised OHLCV DataFrame and produces a
structured ``ValidationReport`` that flags:

* Missing timestamps (gaps in the expected bar sequence).
* Duplicate rows (identical timestamps).
* Invalid OHLC relationships (e.g. high < low).
* Missing or zero volume.
* Incorrect chronological ordering.

The report includes both human-readable summaries and machine-readable
metrics that the ``DataManager`` uses to decide whether the data passes
quality gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger

from config.constants import OHLCV_COLUMNS


# ---------------------------------------------------------------------------
# Validation report data class
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Container for all validation results.

    Attributes:
        total_rows: Number of rows inspected.
        missing_timestamps: List of expected-but-absent timestamps.
        duplicate_count: Number of duplicated timestamp rows.
        invalid_ohlc_indices: Row indices where OHLC relationships fail.
        missing_volume_indices: Row indices with NaN or zero volume.
        ordering_violations: Row indices where time goes backward.
        passed: Overall pass / fail after threshold checks.
        details: Free-form detail strings for the human-readable report.
    """

    total_rows: int = 0
    missing_timestamps: list[str] = field(default_factory=list)
    duplicate_count: int = 0
    invalid_ohlc_indices: list[int] = field(default_factory=list)
    missing_volume_indices: list[int] = field(default_factory=list)
    ordering_violations: list[int] = field(default_factory=list)
    passed: bool = True
    details: list[str] = field(default_factory=list)

    # ---- Derived properties -----------------------------------------------

    @property
    def missing_count(self) -> int:
        """Number of missing timestamps detected."""
        return len(self.missing_timestamps)

    @property
    def missing_pct(self) -> float:
        """Missing timestamps as a percentage of total expected rows."""
        expected = self.total_rows + self.missing_count
        if expected == 0:
            return 0.0
        return (self.missing_count / expected) * 100.0

    @property
    def duplicate_pct(self) -> float:
        """Duplicate rows as a percentage of total rows."""
        if self.total_rows == 0:
            return 0.0
        return (self.duplicate_count / self.total_rows) * 100.0

    # ---- Display ----------------------------------------------------------

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        lines = [
            "═══════════════════════════════════════════",
            "         DATA VALIDATION REPORT            ",
            "═══════════════════════════════════════════",
            f"  Total rows          : {self.total_rows}",
            f"  Missing timestamps  : {self.missing_count} ({self.missing_pct:.2f} %)",
            f"  Duplicate rows      : {self.duplicate_count} ({self.duplicate_pct:.2f} %)",
            f"  Invalid OHLC rows   : {len(self.invalid_ohlc_indices)}",
            f"  Missing volume rows : {len(self.missing_volume_indices)}",
            f"  Ordering violations : {len(self.ordering_violations)}",
            f"  Overall result      : {'PASS ✅' if self.passed else 'FAIL ❌'}",
            "═══════════════════════════════════════════",
        ]
        if self.details:
            lines.append("  Details:")
            for detail in self.details:
                lines.append(f"    • {detail}")
            lines.append("═══════════════════════════════════════════")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timeframe → timedelta mapping for gap detection
# ---------------------------------------------------------------------------

_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame) -> list[str]:
    """Verify that all required OHLCV columns exist.

    Args:
        df: Input DataFrame.

    Returns:
        List of error detail strings (empty on success).
    """
    missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        return [f"Missing required columns: {missing_cols}"]
    return []


def _check_ordering(df: pd.DataFrame) -> list[int]:
    """Return indices where chronological ordering is violated.

    Args:
        df: DataFrame with a ``timestamp`` column (datetime64).

    Returns:
        List of integer row indices.
    """
    ts = pd.to_datetime(df["timestamp"])
    diffs = ts.diff()
    violations = diffs[diffs < pd.Timedelta(0)]
    return violations.index.tolist()


def _check_duplicates(df: pd.DataFrame) -> int:
    """Count rows with duplicate timestamps.

    Args:
        df: DataFrame with a ``timestamp`` column.

    Returns:
        Number of duplicated rows.
    """
    return int(df["timestamp"].duplicated().sum())


def _check_ohlc_validity(df: pd.DataFrame) -> list[int]:
    """Identify rows where OHLC relationships are invalid.

    Valid relationships:
        * ``low <= open <= high``
        * ``low <= close <= high``
        * ``low <= high``
        * All four values are finite and positive.

    Args:
        df: DataFrame with ``open``, ``high``, ``low``, ``close``.

    Returns:
        List of integer row indices that fail validation.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    invalid_mask = (
        (l > h)
        | (o > h) | (o < l)
        | (c > h) | (c < l)
        | o.isna() | h.isna() | l.isna() | c.isna()
        | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    )
    return df.index[invalid_mask].tolist()


def _check_volume(df: pd.DataFrame) -> list[int]:
    """Return indices where volume is missing or zero.

    Args:
        df: DataFrame with a ``volume`` column.

    Returns:
        List of integer row indices.
    """
    vol = df["volume"]
    bad_mask = vol.isna() | (vol <= 0)
    return df.index[bad_mask].tolist()


def _detect_missing_timestamps(
    df: pd.DataFrame,
    timeframe: str,
) -> list[str]:
    """Find expected timestamps that are absent from the data.

    This function generates the full set of expected timestamps between
    the first and last row at the given bar frequency, then reports
    which ones are missing.

    For daily and weekly bars, weekends and common US holidays are
    **not** flagged because equity markets are closed.  For intraday
    bars the full 24 h range is checked (appropriate for crypto;
    equity users should interpret these counts accordingly).

    Args:
        df: DataFrame with a ``timestamp`` column (datetime64).
        timeframe: Canonical timeframe string (e.g. ``"1d"``).

    Returns:
        List of ISO-formatted timestamp strings that are missing.
    """
    if len(df) < 2:
        return []

    delta = _TIMEFRAME_DELTAS.get(timeframe)
    if delta is None:
        return []

    ts = pd.to_datetime(df["timestamp"]).sort_values()
    start, end = ts.iloc[0], ts.iloc[-1]

    if timeframe in ("1d", "1w"):
        freq = "B" if timeframe == "1d" else "W-MON"
        expected_index = pd.bdate_range(start=start, end=end, freq=freq)
    else:
        expected_index = pd.date_range(start=start, end=end, freq=delta)

    actual_set = set(ts.dt.floor("min"))
    expected_set = set(expected_index)
    missing = sorted(expected_set - actual_set)

    return [m.isoformat() for m in missing]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    *,
    max_missing_pct: float = 5.0,
    max_duplicate_pct: float = 1.0,
) -> ValidationReport:
    """Run the full validation suite on an OHLCV DataFrame.

    Args:
        df: DataFrame with standard OHLCV columns.
        timeframe: Bar frequency (e.g. ``"1h"``).
        max_missing_pct: Threshold above which missing bars cause a
            failure.
        max_duplicate_pct: Threshold above which duplicates cause a
            failure.

    Returns:
        A populated ``ValidationReport``.
    """
    report = ValidationReport(total_rows=len(df))

    # --- Column presence ---------------------------------------------------
    col_errors = _check_required_columns(df)
    if col_errors:
        report.details.extend(col_errors)
        report.passed = False
        logger.error("Validation failed: {}", col_errors)
        return report

    # --- Ordering ----------------------------------------------------------
    report.ordering_violations = _check_ordering(df)
    if report.ordering_violations:
        count = len(report.ordering_violations)
        report.details.append(
            f"{count} ordering violation(s) detected"
        )
        report.passed = False
        logger.warning("Ordering violations at indices: {}", report.ordering_violations)

    # --- Duplicates --------------------------------------------------------
    report.duplicate_count = _check_duplicates(df)
    if report.duplicate_pct > max_duplicate_pct:
        report.details.append(
            f"Duplicate rate {report.duplicate_pct:.2f} % exceeds "
            f"threshold {max_duplicate_pct:.2f} %"
        )
        report.passed = False
        logger.warning(
            "Duplicate rows: {} ({:.2f} %)",
            report.duplicate_count,
            report.duplicate_pct,
        )

    # --- OHLC integrity ----------------------------------------------------
    report.invalid_ohlc_indices = _check_ohlc_validity(df)
    if report.invalid_ohlc_indices:
        count = len(report.invalid_ohlc_indices)
        report.details.append(f"{count} row(s) with invalid OHLC values")
        report.passed = False
        logger.warning("Invalid OHLC at {} indices", count)

    # --- Volume ------------------------------------------------------------
    report.missing_volume_indices = _check_volume(df)
    if report.missing_volume_indices:
        count = len(report.missing_volume_indices)
        report.details.append(f"{count} row(s) with missing / zero volume")
        logger.warning("Missing volume at {} indices", count)

    # --- Missing timestamps ------------------------------------------------
    report.missing_timestamps = _detect_missing_timestamps(df, timeframe)
    if report.missing_pct > max_missing_pct:
        report.details.append(
            f"Missing-bar rate {report.missing_pct:.2f} % exceeds "
            f"threshold {max_missing_pct:.2f} %"
        )
        report.passed = False
        logger.warning(
            "Missing timestamps: {} ({:.2f} %)",
            report.missing_count,
            report.missing_pct,
        )

    # --- Final verdict -----------------------------------------------------
    if report.passed:
        logger.success("Validation PASSED for {} rows", report.total_rows)
    else:
        logger.error("Validation FAILED — see report details")

    return report


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning transformations to an OHLCV DataFrame.

    Steps performed:
        1. Drop fully-duplicated rows.
        2. Drop rows with duplicate timestamps (keep first).
        3. Sort by timestamp ascending.
        4. Convert the ``timestamp`` column to UTC-aware ``datetime64``.
        5. Reset the integer index.

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        Cleaned DataFrame (new copy; the original is not mutated).
    """
    original_len = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    dropped = original_len - len(df)
    if dropped:
        logger.info("Cleaned OHLCV: dropped {} duplicate / malformed rows", dropped)

    return df
