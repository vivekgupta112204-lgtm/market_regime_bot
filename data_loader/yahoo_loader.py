"""Yahoo Finance data loader.

Uses the ``yfinance`` library to download historical OHLCV data.
Handles the quirks of Yahoo's API:

* Renames columns to the canonical schema.
* Resets the index so ``Date`` becomes a regular column.
* Applies timezone normalisation to UTC.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.constants import OHLCV_COLUMNS, YAHOO_TIMEFRAME_MAP


class YahooLoader:
    """Download OHLCV bars from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g. ``"SPY"``).
        timeframe: Canonical timeframe string (e.g. ``"1d"``).
        max_retries: Maximum download attempts.
        retry_delay: Initial retry back-off in seconds.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        max_retries: int = 3,
        retry_delay: int = 5,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        yf_interval = YAHOO_TIMEFRAME_MAP.get(timeframe)
        if yf_interval is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}' for Yahoo Finance. "
                f"Supported: {list(YAHOO_TIMEFRAME_MAP)}"
            )
        self._yf_interval = yf_interval

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _download(
        self,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Execute a single yfinance download with retry logic.

        Args:
            start: Start date string ``YYYY-MM-DD``.
            end: End date string ``YYYY-MM-DD``.

        Returns:
            Raw DataFrame from ``yfinance``.

        Raises:
            ValueError: When yfinance returns an empty DataFrame.
        """
        logger.info(
            "Yahoo Finance: downloading {} {} from {} to {}",
            self.symbol,
            self._yf_interval,
            start,
            end,
        )
        ticker = yf.Ticker(self.symbol)
        df: pd.DataFrame = ticker.history(
            start=start,
            end=end,
            interval=self._yf_interval,
            auto_adjust=True,
        )

        if df.empty:
            raise ValueError(
                f"Yahoo Finance returned no data for {self.symbol} "
                f"({start} → {end}, interval={self._yf_interval}). "
                "Verify the symbol is valid and the date range is correct."
            )

        return df

    def fetch(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Download and normalise OHLCV data.

        Args:
            start_date: Start of the data window.
            end_date: End of the data window.

        Returns:
            A DataFrame with columns matching ``OHLCV_COLUMNS``.
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        raw = self._download(start_str, end_str)

        # yfinance returns a DatetimeIndex — flatten it.
        raw = raw.reset_index()

        # Rename columns to canonical names.
        rename_map: dict[str, str] = {
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        raw = raw.rename(columns=rename_map)

        # Ensure UTC.
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

        # Keep only canonical columns.
        available = [c for c in OHLCV_COLUMNS if c in raw.columns]
        df = raw[available].copy()

        logger.success(
            "Yahoo Finance: fetched {} rows for {}", len(df), self.symbol
        )
        return df
