"""Real-time data service for regime detection.

Provides functions to fetch the latest OHLCV candles from the
configured broker so the detection engine can classify the current
market state.

This module reuses the Phase 1 data loaders but wraps them with
convenience methods tailored to the real-time detection use case:

* Fetch the last *N* candles (default 200 for warm-up).
* Return a standardised DataFrame ready for the feature pipeline.
* Handle errors gracefully with logging.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from config.constants import Broker
from config.settings import Settings
from data_loader.alpaca_loader import AlpacaLoader
from data_loader.binance_loader import BinanceLoader
from data_loader.bybit_loader import BybitLoader
from data_loader.yahoo_loader import YahooLoader
from utils.validation import clean_ohlcv


# Number of bars the feature pipeline needs to compute all rolling
# indicators without producing NaN rows.
_WARMUP_BARS: int = 200


def fetch_latest_candles(
    settings: Settings,
    lookback_bars: int = _WARMUP_BARS,
) -> pd.DataFrame:
    """Download the most recent candles from the configured broker.

    The function calculates an appropriate start date based on the
    requested *lookback_bars* and the configured timeframe, then
    downloads up to the current moment.

    Args:
        settings: Application settings (broker, symbol, timeframe, keys).
        lookback_bars: Number of historical bars to fetch.  Must be
            large enough to cover the feature-engineering warm-up
            period.

    Returns:
        A cleaned OHLCV DataFrame with UTC timestamps, ready for
        feature engineering.

    Raises:
        ValueError: When the broker returns no data.
    """
    # Estimate the calendar span needed for *lookback_bars*.
    tf_to_days: dict[str, float] = {
        "1m": 1 / 1440,
        "5m": 5 / 1440,
        "15m": 15 / 1440,
        "30m": 30 / 1440,
        "1h": 1 / 24,
        "4h": 4 / 24,
        "1d": 1.0,
        "1w": 7.0,
    }

    days_per_bar = tf_to_days.get(settings.timeframe.value, 1.0)
    # Add 50 % margin to account for weekends / holidays.
    total_days = int(lookback_bars * days_per_bar * 1.5) + 5
    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=total_days)

    logger.info(
        "Fetching latest candles: {} {} {} (lookback={}d)",
        settings.broker.value,
        settings.symbol,
        settings.timeframe.value,
        total_days,
    )

    loader = _build_loader(settings)
    raw_df = loader.fetch(start_date, end_date)
    df = clean_ohlcv(raw_df)

    # Trim to the last *lookback_bars* rows.
    if len(df) > lookback_bars:
        df = df.tail(lookback_bars).reset_index(drop=True)

    logger.info(
        "Fetched {} candles ({} → {})",
        len(df),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
    )

    return df


def _build_loader(
    settings: Settings,
) -> YahooLoader | BinanceLoader | BybitLoader | AlpacaLoader:
    """Instantiate the appropriate exchange loader.

    Args:
        settings: Application settings.

    Returns:
        A loader instance.

    Raises:
        ValueError: When the broker is unsupported.
    """
    s = settings
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
