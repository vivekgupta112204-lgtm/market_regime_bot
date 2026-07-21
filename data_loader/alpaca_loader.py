"""Alpaca Markets data loader.

Uses the official ``alpaca-py`` SDK to retrieve historical bar data
from Alpaca's Market Data API (v2).  Supports both stocks and crypto
via the unified ``StockHistoricalDataClient`` / ``CryptoHistoricalDataClient``.

This loader defaults to the **stock** client.  For crypto pairs,
extend or configure as needed in Phase 2.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.constants import ALPACA_TIMEFRAME_MAP, OHLCV_COLUMNS

# Alpaca SDK imports — guarded so the rest of the codebase works even
# when alpaca-py is not installed (other brokers remain available).
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


# Mapping from our canonical labels to Alpaca ``TimeFrame`` objects.
_TF_MAP: dict[str, "TimeFrame"] = {}
if _ALPACA_AVAILABLE:
    _TF_MAP = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
        "1w": TimeFrame(1, TimeFrameUnit.Week),
    }


class AlpacaLoader:
    """Download OHLCV bars from Alpaca Markets.

    Args:
        symbol: Ticker symbol (e.g. ``"SPY"``).
        timeframe: Canonical timeframe string (e.g. ``"1d"``).
        api_key: Alpaca API key.
        secret_key: Alpaca secret key.
        base_url: Alpaca REST endpoint (paper or live).
        max_retries: Maximum download attempts.
        retry_delay: Initial retry back-off in seconds.

    Raises:
        ImportError: When ``alpaca-py`` is not installed.
        ValueError: When the timeframe is not supported.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        max_retries: int = 3,
        retry_delay: int = 5,
    ) -> None:
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is not installed. "
                "Run: pip install alpaca-py"
            )

        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if timeframe not in _TF_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}' for Alpaca. "
                f"Supported: {list(ALPACA_TIMEFRAME_MAP)}"
            )
        self._alpaca_tf: TimeFrame = _TF_MAP[timeframe]

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca requires ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "to be set in the .env file."
            )

        self._client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _download(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Execute a single Alpaca bars request with retry logic.

        Args:
            start: Start datetime (UTC).
            end: End datetime (UTC).

        Returns:
            DataFrame produced by the Alpaca SDK.

        Raises:
            ValueError: When Alpaca returns no data.
        """
        logger.info(
            "Alpaca: downloading {} {} from {} to {}",
            self.symbol,
            self.timeframe,
            start.date(),
            end.date(),
        )
        request = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=self._alpaca_tf,
            start=start,
            end=end,
        )

        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            raise ValueError(
                f"Alpaca returned no data for {self.symbol} "
                f"({start.date()} → {end.date()})."
            )
        return df

    def fetch(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Download and normalise OHLCV data from Alpaca.

        Args:
            start_date: Start of the data window.
            end_date: End of the data window.

        Returns:
            A DataFrame with columns matching ``OHLCV_COLUMNS``.
        """
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

        raw = self._download(start_dt, end_dt)

        # Alpaca returns a MultiIndex (symbol, timestamp) — flatten.
        raw = raw.reset_index()

        rename_map: dict[str, str] = {
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        raw = raw.rename(columns=rename_map)

        # Ensure UTC.
        if "timestamp" in raw.columns:
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

        available = [c for c in OHLCV_COLUMNS if c in raw.columns]
        df = raw[available].copy()

        logger.success(
            "Alpaca: fetched {} rows for {}", len(df), self.symbol
        )
        return df
