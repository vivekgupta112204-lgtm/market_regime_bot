"""General-purpose helper functions for Market Regime Bot.

This module collects small, stateless utilities that are used across
multiple layers of the application — file-naming conventions, timestamp
conversions, caching logic, and data persistence.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from config.constants import DATE_FORMAT


# ---------------------------------------------------------------------------
# File-naming helpers
# ---------------------------------------------------------------------------

def build_filename(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    broker: str,
) -> str:
    """Construct a deterministic, filesystem-safe base filename.

    The name encodes every parameter that makes a dataset unique so
    cached artefacts can be looked up without ambiguity.

    Args:
        symbol: Instrument ticker (e.g. ``"BTCUSDT"``).
        timeframe: Bar size (e.g. ``"1d"``).
        start_date: Start date as ``YYYY-MM-DD``.
        end_date: End date as ``YYYY-MM-DD``.
        broker: Data-source identifier (e.g. ``"binance"``).

    Returns:
        Filename string **without** extension (e.g.
        ``"binance_BTCUSDT_1d_20200101_20250101"``).
    """
    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    return f"{broker}_{symbol}_{timeframe}_{safe_start}_{safe_end}"


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex-digest of a file's contents.

    Args:
        path: Path to the target file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def to_utc_timestamp(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware and in UTC.

    Args:
        dt: A ``datetime`` instance (naïve or aware).

    Returns:
        The same instant expressed in UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ms_to_utc_datetime(epoch_ms: int) -> datetime:
    """Convert a Unix-epoch millisecond timestamp to a UTC datetime.

    Args:
        epoch_ms: Milliseconds since 1970-01-01 00:00:00 UTC.

    Returns:
        A timezone-aware ``datetime`` in UTC.
    """
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def is_cache_valid(cache_path: Path, expiry_hours: int) -> bool:
    """Check whether a cached file exists and is still fresh.

    Args:
        cache_path: Path to the cached file.
        expiry_hours: Number of hours after which the cache is stale.

    Returns:
        ``True`` when the file exists and was modified fewer than
        *expiry_hours* ago.
    """
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0
    fresh = age_hours < expiry_hours
    if fresh:
        logger.info("Cache hit: {} (age {:.1f} h)", cache_path.name, age_hours)
    else:
        logger.info("Cache stale: {} (age {:.1f} h)", cache_path.name, age_hours)
    return fresh


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_dataframe(
    df: pd.DataFrame,
    directory: Path,
    base_name: str,
    *,
    csv: bool = True,
    parquet: bool = True,
) -> dict[str, Path]:
    """Persist a DataFrame to CSV and / or Parquet.

    Args:
        df: DataFrame to save.
        directory: Target directory (created if missing).
        base_name: Filename stem (no extension).
        csv: Whether to write a ``.csv`` copy.
        parquet: Whether to write a ``.parquet`` copy.

    Returns:
        Mapping of format name → written file path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if csv:
        csv_path = directory / f"{base_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved CSV  → {} ({} rows)", csv_path.name, len(df))
        paths["csv"] = csv_path

    if parquet:
        parquet_path = directory / f"{base_name}.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        logger.info("Saved Parquet → {} ({} rows)", parquet_path.name, len(df))
        paths["parquet"] = parquet_path

    return paths


def load_cached_dataframe(cache_path: Path) -> pd.DataFrame:
    """Load a Parquet file from cache and return it as a DataFrame.

    Args:
        cache_path: Path to a ``.parquet`` file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If *cache_path* does not exist.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    df = pd.read_parquet(cache_path, engine="pyarrow")
    logger.info("Loaded cache → {} ({} rows)", cache_path.name, len(df))
    return df


def format_date(d: datetime | pd.Timestamp | str, fmt: str = DATE_FORMAT) -> str:
    """Format a date-like object into a standardised string.

    Args:
        d: Date value — can be a ``datetime``, pandas ``Timestamp``, or
           an already-formatted string (returned unchanged).
        fmt: ``strftime`` format string.

    Returns:
        Formatted date string.
    """
    if isinstance(d, str):
        return d
    return d.strftime(fmt)
