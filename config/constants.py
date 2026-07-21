"""Application-wide constants for Market Regime Bot.

Centralises magic strings, enumerations, and default values so they
are defined in exactly one place across the entire codebase.
"""

from enum import Enum


class Broker(str, Enum):
    """Supported data-source / broker identifiers."""

    YAHOO = "yahoo"
    BINANCE = "binance"
    BYBIT = "bybit"
    ALPACA = "alpaca"


class Timeframe(str, Enum):
    """Canonical timeframe labels understood by every loader."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


# Mapping from canonical timeframe to Yahoo Finance interval strings.
YAHOO_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "1h",     # Yahoo lacks 4 h — fetch 1 h, resample downstream
    "1d": "1d",
    "1w": "1wk",
}

# CCXT uses its own timeframe notation (shared by Binance & Bybit).
CCXT_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}

# Alpaca timeframe mapping.
ALPACA_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
    "1w": "1Week",
}

# Standard OHLCV column order used everywhere in the pipeline.
OHLCV_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Maximum number of candles Binance / Bybit return per single API call.
CCXT_FETCH_LIMIT: int = 1000

# Default date format used for file naming and display.
DATE_FORMAT: str = "%Y-%m-%d"

# ISO-8601 datetime format with timezone.
DATETIME_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"


# ===================================================================
# Phase 2 — Feature Engineering & HMM Training constants
# ===================================================================

class ScalerType(str, Enum):
    """Supported feature-scaling strategies."""

    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"


class CovarianceType(str, Enum):
    """Covariance matrix types accepted by hmmlearn."""

    FULL = "full"
    TIED = "tied"
    DIAG = "diag"
    SPHERICAL = "spherical"


# Default indicator look-back periods.
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
EMA_SHORT: int = 20
EMA_LONG: int = 50
SMA_SHORT: int = 20
SMA_LONG: int = 50
BB_PERIOD: int = 20
BB_STD: float = 2.0
STOCH_PERIOD: int = 14
CCI_PERIOD: int = 20
MOMENTUM_PERIOD: int = 10
ROC_PERIOD: int = 10
VOLATILITY_WINDOW: int = 20
DONCHIAN_PERIOD: int = 20
