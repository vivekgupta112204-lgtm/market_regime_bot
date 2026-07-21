"""Technical-indicator feature engineering for OHLCV data.

Computes 26+ features from raw OHLCV bars.  Every function is a pure
transform — it receives a DataFrame and returns an augmented copy —
making the module trivially testable and composable.

Feature categories
~~~~~~~~~~~~~~~~~~
* **Returns** — daily return, log return
* **Volatility** — rolling volatility, ATR, Bollinger width
* **Trend** — EMA, SMA, EMA distance, ADX, MACD suite
* **Momentum** — RSI, Stochastic, CCI, momentum, ROC
* **Volume** — OBV, volume change
* **Candle anatomy** — high-low range, body size, wicks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.constants import (
    ATR_PERIOD,
    ADX_PERIOD,
    BB_PERIOD,
    BB_STD,
    CCI_PERIOD,
    EMA_LONG,
    EMA_SHORT,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MOMENTUM_PERIOD,
    ROC_PERIOD,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
    STOCH_PERIOD,
    VOLATILITY_WINDOW,
    DONCHIAN_PERIOD,
)


# ===================================================================
# Return features
# ===================================================================

def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage daily return.

    Args:
        df: DataFrame with a ``close`` column.

    Returns:
        DataFrame with ``daily_return`` column appended.
    """
    df["daily_return"] = df["close"].pct_change()
    return df


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add logarithmic daily return.

    Args:
        df: DataFrame with a ``close`` column.

    Returns:
        DataFrame with ``log_return`` column appended.
    """
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


# ===================================================================
# Volatility features
# ===================================================================

def add_rolling_volatility(
    df: pd.DataFrame, window: int = VOLATILITY_WINDOW
) -> pd.DataFrame:
    """Add rolling standard deviation of log returns.

    Args:
        df: DataFrame with a ``log_return`` column.
        window: Look-back window.

    Returns:
        DataFrame with ``rolling_volatility`` column appended.
    """
    if "log_return" not in df.columns:
        df = add_log_return(df)
    df["rolling_volatility"] = df["log_return"].rolling(window=window).std()
    return df


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Add Average True Range (ATR).

    Args:
        df: DataFrame with ``high``, ``low``, ``close`` columns.
        period: Look-back period.

    Returns:
        DataFrame with ``atr`` column appended.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=period).mean()
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = BB_PERIOD,
    std_dev: float = BB_STD,
) -> pd.DataFrame:
    """Add Bollinger Bands and bandwidth.

    Args:
        df: DataFrame with a ``close`` column.
        period: Moving-average look-back.
        std_dev: Number of standard deviations for the bands.

    Returns:
        DataFrame with ``bb_upper``, ``bb_lower``, ``bb_width`` columns.
    """
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    return df


# ===================================================================
# Trend features
# ===================================================================

def add_ema(
    df: pd.DataFrame,
    short: int = EMA_SHORT,
    long: int = EMA_LONG,
) -> pd.DataFrame:
    """Add exponential moving averages and their distance.

    Args:
        df: DataFrame with a ``close`` column.
        short: Short EMA span.
        long: Long EMA span.

    Returns:
        DataFrame with ``ema_20``, ``ema_50``, ``ema_distance`` columns.
    """
    df["ema_20"] = df["close"].ewm(span=short, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=long, adjust=False).mean()
    df["ema_distance"] = (df["ema_20"] - df["ema_50"]) / df["close"]
    return df


def add_sma(
    df: pd.DataFrame,
    short: int = SMA_SHORT,
    long: int = SMA_LONG,
) -> pd.DataFrame:
    """Add simple moving averages.

    Args:
        df: DataFrame with a ``close`` column.
        short: Short SMA window.
        long: Long SMA window.

    Returns:
        DataFrame with ``sma_20``, ``sma_50`` columns.
    """
    df["sma_20"] = df["close"].rolling(window=short).mean()
    df["sma_50"] = df["close"].rolling(window=long).mean()
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """Add MACD line, signal line, and histogram.

    Args:
        df: DataFrame with a ``close`` column.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.

    Returns:
        DataFrame with ``macd``, ``macd_signal``, ``macd_histogram``
        columns.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    return df


def add_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Add Average Directional Index (ADX).

    Uses the classic Wilder smoothing method.

    Args:
        df: DataFrame with ``high``, ``low``, ``close`` columns.
        period: Look-back period.

    Returns:
        DataFrame with ``adx`` column appended.
    """
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Wilder smoothing (EWM with alpha=1/period)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )

    dx = (((plus_di - minus_di).abs()) / (plus_di + minus_di)) * 100
    df["adx"] = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return df


# ===================================================================
# Momentum features
# ===================================================================

def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Add Relative Strength Index (RSI).

    Args:
        df: DataFrame with a ``close`` column.
        period: Look-back period.

    Returns:
        DataFrame with ``rsi`` column appended.
    """
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(
    df: pd.DataFrame, period: int = STOCH_PERIOD
) -> pd.DataFrame:
    """Add Stochastic Oscillator (%K).

    Args:
        df: DataFrame with ``high``, ``low``, ``close`` columns.
        period: Look-back period.

    Returns:
        DataFrame with ``stochastic_k`` column appended.
    """
    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    df["stochastic_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    return df


def add_cci(df: pd.DataFrame, period: int = CCI_PERIOD) -> pd.DataFrame:
    """Add Commodity Channel Index (CCI).

    Args:
        df: DataFrame with ``high``, ``low``, ``close`` columns.
        period: Look-back period.

    Returns:
        DataFrame with ``cci`` column appended.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    df["cci"] = (typical_price - sma) / (0.015 * mad)
    return df


def add_momentum(
    df: pd.DataFrame, period: int = MOMENTUM_PERIOD
) -> pd.DataFrame:
    """Add raw momentum (price difference over *period* bars).

    Args:
        df: DataFrame with a ``close`` column.
        period: Look-back period.

    Returns:
        DataFrame with ``momentum`` column appended.
    """
    df["momentum"] = df["close"] - df["close"].shift(period)
    return df


def add_roc(df: pd.DataFrame, period: int = ROC_PERIOD) -> pd.DataFrame:
    """Add Rate of Change (ROC).

    Args:
        df: DataFrame with a ``close`` column.
        period: Look-back period.

    Returns:
        DataFrame with ``roc`` column appended.
    """
    df["roc"] = df["close"].pct_change(periods=period) * 100
    return df


# ===================================================================
# Volume features
# ===================================================================

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume (OBV).

    Args:
        df: DataFrame with ``close`` and ``volume`` columns.

    Returns:
        DataFrame with ``obv`` column appended.
    """
    direction = np.sign(df["close"].diff())
    df["obv"] = (direction * df["volume"]).fillna(0).cumsum()
    return df


def add_volume_change(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage volume change.

    Args:
        df: DataFrame with a ``volume`` column.

    Returns:
        DataFrame with ``volume_change`` column appended.
    """
    df["volume_change"] = df["volume"].pct_change()
    return df


# ===================================================================
# Candle anatomy features
# ===================================================================

def add_donchian_channels(df: pd.DataFrame, period: int = DONCHIAN_PERIOD) -> pd.DataFrame:
    """Add Donchian Channel upper and lower bands.

    Args:
        df: DataFrame with ``high`` and ``low`` columns.
        period: Look-back period.

    Returns:
        DataFrame with ``donchian_upper`` and ``donchian_lower`` appended.
    """
    df["donchian_upper"] = df["high"].rolling(window=period).max()
    df["donchian_lower"] = df["low"].rolling(window=period).min()
    return df



def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candle-body and wick metrics.

    Args:
        df: DataFrame with ``open``, ``high``, ``low``, ``close``.

    Returns:
        DataFrame with ``high_low_range``, ``candle_body``,
        ``upper_wick``, ``lower_wick`` columns appended.
    """
    df["high_low_range"] = df["high"] - df["low"]
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    return df


# ===================================================================
# Orchestrator
# ===================================================================

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature-engineering function to the DataFrame.

    The original OHLCV columns are preserved; computed columns are
    appended.  Rows containing NaN values introduced by rolling
    indicators are dropped at the end.

    Args:
        df: Cleaned OHLCV DataFrame (must contain at minimum
            ``timestamp``, ``open``, ``high``, ``low``, ``close``,
            ``volume``).

    Returns:
        Feature-enriched DataFrame with NaN rows removed.
    """
    logger.info("Computing features on {} rows …", len(df))

    df = df.copy()

    # Returns
    df = add_daily_return(df)
    df = add_log_return(df)

    # Volatility
    df = add_rolling_volatility(df)
    df = add_atr(df)
    df = add_bollinger_bands(df)

    # Trend
    df = add_ema(df)
    df = add_sma(df)
    df = add_macd(df)
    df = add_adx(df)

    # Momentum
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_cci(df)
    df = add_momentum(df)
    df = add_roc(df)

    # Volume
    df = add_obv(df)
    df = add_volume_change(df)

    # Candle anatomy
    df = add_candle_features(df)
    df = add_donchian_channels(df)

    # Drop warm-up NaN rows produced by rolling windows
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = rows_before - len(df)
    logger.info(
        "Features computed: {} total columns, {} warm-up rows dropped, "
        "{} rows remaining",
        len(df.columns),
        dropped,
        len(df),
    )

    return df
