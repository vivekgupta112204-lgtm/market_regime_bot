"""Binance data loader via CCXT.

Downloads historical OHLCV data from Binance using the unified CCXT
interface.  Because the exchange caps each request at ~1 000 candles,
this loader **paginates automatically** from ``start_date`` to
``end_date``.

Authentication is optional — public endpoints are used for OHLCV.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from time import sleep

import ccxt
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.constants import CCXT_FETCH_LIMIT, CCXT_TIMEFRAME_MAP, OHLCV_COLUMNS


class BinanceLoader:
    """Paginated OHLCV downloader for Binance.

    Args:
        symbol: Market pair (e.g. ``"BTC/USDT"``).
        timeframe: Canonical timeframe string (e.g. ``"1h"``).
        api_key: Optional Binance API key.
        secret_key: Optional Binance secret key.
        max_retries: Maximum per-page retry attempts.
        retry_delay: Initial retry back-off in seconds.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        api_key: str = "",
        secret_key: str = "",
        max_retries: int = 3,
        retry_delay: int = 5,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        ccxt_tf = CCXT_TIMEFRAME_MAP.get(timeframe)
        if ccxt_tf is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}' for Binance. "
                f"Supported: {list(CCXT_TIMEFRAME_MAP)}"
            )
        self._ccxt_tf = ccxt_tf

        exchange_config: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if api_key and secret_key:
            exchange_config["apiKey"] = api_key
            exchange_config["secret"] = secret_key

        self._exchange = ccxt.binance(exchange_config)

    @retry(
        retry=retry_if_exception_type(
            (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _fetch_page(self, since_ms: int) -> list[list]:
        """Fetch a single page of OHLCV candles.

        Args:
            since_ms: Unix epoch millisecond timestamp for the page start.

        Returns:
            List of ``[timestamp_ms, open, high, low, close, volume]``
            rows returned by CCXT.

        Raises:
            ccxt.BadSymbol: When the symbol is not listed on Binance.
            ccxt.AuthenticationError: When credentials are invalid.
        """
        return self._exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self._ccxt_tf,
            since=since_ms,
            limit=CCXT_FETCH_LIMIT,
        )

    def fetch(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Download the full date range via automatic pagination.

        Args:
            start_date: Start of the data window (UTC).
            end_date: End of the data window (UTC).

        Returns:
            A DataFrame with columns matching ``OHLCV_COLUMNS``.

        Raises:
            ValueError: When no data is returned for the entire range.
        """
        start_ms = int(
            datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        end_ms = int(
            datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

        all_candles: list[list] = []
        current_ms = start_ms
        page_count = 0

        logger.info(
            "Binance: downloading {} {} from {} to {}",
            self.symbol,
            self._ccxt_tf,
            start_date,
            end_date,
        )

        while current_ms < end_ms:
            page = self._fetch_page(current_ms)
            if not page:
                break

            all_candles.extend(page)
            page_count += 1
            last_ts = page[-1][0]

            logger.debug(
                "Binance page {}: {} candles (last ts={})",
                page_count,
                len(page),
                last_ts,
            )

            # Advance cursor past the last received candle.
            current_ms = last_ts + 1

            # Respect exchange rate limits.
            sleep(self._exchange.rateLimit / 1000.0)

        if not all_candles:
            raise ValueError(
                f"Binance returned no data for {self.symbol} "
                f"({start_date} → {end_date}, tf={self._ccxt_tf})."
            )

        df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Trim to requested range.
        end_dt = pd.Timestamp(end_date, tz="UTC")
        df = df[df["timestamp"] < end_dt].reset_index(drop=True)

        logger.success(
            "Binance: fetched {} rows across {} pages for {}",
            len(df),
            page_count,
            self.symbol,
        )
        return df
