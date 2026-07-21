"""Unified data-management façade for Market Regime Bot.

``DataManager`` is the single entry-point that the rest of the
application uses to obtain clean, validated OHLCV data.  It:

1. Checks the on-disk cache first.
2. If no valid cache exists, delegates to the appropriate exchange
   loader.
3. Cleans and validates the downloaded data.
4. Persists the result as CSV + Parquet.
5. Returns the ready-to-use DataFrame.

This design means callers never need to know *which* broker is being
used — they simply call ``manager.get_data()`` and receive a uniform
DataFrame.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from loguru import logger

from config.constants import Broker
from config.settings import Settings
from data_loader.alpaca_loader import AlpacaLoader
from data_loader.binance_loader import BinanceLoader
from data_loader.bybit_loader import BybitLoader
from data_loader.yahoo_loader import YahooLoader
from utils.helpers import (
    build_filename,
    is_cache_valid,
    load_cached_dataframe,
    save_dataframe,
)
from utils.validation import ValidationReport, clean_ohlcv, validate_ohlcv

LoaderType = Union[YahooLoader, BinanceLoader, BybitLoader, AlpacaLoader]


class DataManager:
    """Orchestrate data download, caching, cleaning, and validation.

    Args:
        settings: Fully validated ``Settings`` instance.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loader: LoaderType = self._build_loader()

    # ------------------------------------------------------------------
    # Loader factory
    # ------------------------------------------------------------------

    def _build_loader(self) -> LoaderType:
        """Instantiate the appropriate exchange loader.

        Returns:
            A loader instance matching ``self._settings.broker``.

        Raises:
            ValueError: When the configured broker is not recognised.
        """
        s = self._settings
        broker = s.broker

        if broker == Broker.YAHOO:
            return YahooLoader(
                symbol=s.symbol,
                timeframe=s.timeframe.value,
                max_retries=s.max_retries,
                retry_delay=s.retry_delay_seconds,
            )

        if broker == Broker.BINANCE:
            return BinanceLoader(
                symbol=s.symbol,
                timeframe=s.timeframe.value,
                api_key=s.api_keys.binance_api_key,
                secret_key=s.api_keys.binance_secret_key,
                max_retries=s.max_retries,
                retry_delay=s.retry_delay_seconds,
            )

        if broker == Broker.BYBIT:
            return BybitLoader(
                symbol=s.symbol,
                timeframe=s.timeframe.value,
                api_key=s.api_keys.bybit_api_key,
                secret_key=s.api_keys.bybit_secret_key,
                max_retries=s.max_retries,
                retry_delay=s.retry_delay_seconds,
            )

        if broker == Broker.ALPACA:
            return AlpacaLoader(
                symbol=s.symbol,
                timeframe=s.timeframe.value,
                api_key=s.api_keys.alpaca_api_key,
                secret_key=s.api_keys.alpaca_secret_key,
                base_url=s.api_keys.alpaca_base_url,
                max_retries=s.max_retries,
                retry_delay=s.retry_delay_seconds,
            )

        raise ValueError(f"Unsupported broker: {broker}")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> "pd.io.common.Path":
        """Build the expected Parquet cache file path."""
        from pathlib import Path

        s = self._settings
        base = build_filename(
            symbol=s.symbol,
            timeframe=s.timeframe.value,
            start_date=str(s.start_date),
            end_date=str(s.end_date),
            broker=s.broker.value,
        )
        return Path(s.cache_dir / f"{base}.parquet")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_data(self) -> tuple[pd.DataFrame, ValidationReport]:
        """Obtain clean, validated OHLCV data.

        The method first checks for a valid cache file.  On a cache miss
        (or when caching is disabled) it downloads fresh data, cleans it,
        validates it, and saves both CSV and Parquet artefacts.

        Returns:
            A 2-tuple of:
            - The cleaned OHLCV DataFrame.
            - The ``ValidationReport`` produced by the validation engine.

        Raises:
            ValueError: When the loader returns empty data.
            Various exchange errors: Propagated after retry exhaustion.
        """
        s = self._settings
        cache_file = self._cache_path()

        # ---- Try cache first ----------------------------------------------
        if s.use_cache and is_cache_valid(cache_file, s.cache_expiry_hours):
            logger.info("Loading data from cache: {}", cache_file.name)
            df = load_cached_dataframe(cache_file)
            report = validate_ohlcv(
                df,
                timeframe=s.timeframe.value,
                max_missing_pct=s.max_missing_pct,
                max_duplicate_pct=s.max_duplicate_pct,
            )
            return df, report

        # ---- Download fresh data ------------------------------------------
        logger.info(
            "Downloading {} data: {} {} ({} → {})",
            s.broker.value,
            s.symbol,
            s.timeframe.value,
            s.start_date,
            s.end_date,
        )

        raw_df = self._loader.fetch(s.start_date, s.end_date)

        # ---- Clean --------------------------------------------------------
        clean_df = clean_ohlcv(raw_df)

        # ---- Validate -----------------------------------------------------
        report = validate_ohlcv(
            clean_df,
            timeframe=s.timeframe.value,
            max_missing_pct=s.max_missing_pct,
            max_duplicate_pct=s.max_duplicate_pct,
        )

        # ---- Persist ------------------------------------------------------
        base = build_filename(
            symbol=s.symbol,
            timeframe=s.timeframe.value,
            start_date=str(s.start_date),
            end_date=str(s.end_date),
            broker=s.broker.value,
        )

        # Save raw copy for auditing.
        save_dataframe(raw_df, s.raw_data_dir, f"{base}_raw")

        # Save cleaned + validated copy.
        save_dataframe(clean_df, s.processed_data_dir, base)

        # Save to cache.
        save_dataframe(clean_df, s.cache_dir, base, csv=False, parquet=True)

        return clean_df, report

    def invalidate_cache(self) -> None:
        """Delete the cached Parquet file for the current configuration."""
        cache_file = self._cache_path()
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Cache invalidated: {}", cache_file.name)
        else:
            logger.debug("No cache file to invalidate.")
