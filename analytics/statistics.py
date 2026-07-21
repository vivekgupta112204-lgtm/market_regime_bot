"""Core performance statistics for backtesting results.

Computes total return, annual return, CAGR, Sharpe, Sortino, Calmar,
profit factor, expectancy, recovery factor, win/loss rate, average
trade, and average holding time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0


def total_return(equity_series: pd.Series) -> float:
    """Calculate total return as a percentage.

    Args:
        equity_series: Time-indexed portfolio value series.

    Returns:
        Total return percentage (e.g. 28.45 for 28.45%).
    """
    if len(equity_series) < 2:
        return 0.0
    start = equity_series.iloc[0]
    end = equity_series.iloc[-1]
    if start <= 0:
        return 0.0
    return ((end - start) / start) * 100.0


def annual_return(equity_series: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate annualised simple return.

    Args:
        equity_series: Time-indexed portfolio value series.
        trading_days: Number of trading days per year.

    Returns:
        Annualised return percentage.
    """
    n_days = len(equity_series)
    if n_days < 2:
        return 0.0
    total = total_return(equity_series) / 100.0
    years = n_days / trading_days
    if years <= 0:
        return 0.0
    return (total / years) * 100.0


def cagr(equity_series: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    Args:
        equity_series: Time-indexed portfolio value series.
        trading_days: Number of trading days per year.

    Returns:
        CAGR as a percentage.
    """
    n_days = len(equity_series)
    if n_days < 2:
        return 0.0
    start = equity_series.iloc[0]
    end = equity_series.iloc[-1]
    if start <= 0:
        return 0.0
    years = n_days / trading_days
    if years <= 0:
        return 0.0
    return (((end / start) ** (1.0 / years)) - 1.0) * 100.0


def sharpe_ratio(
    equity_series: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualised Sharpe ratio.

    Args:
        equity_series: Time-indexed portfolio value series.
        risk_free: Daily risk-free rate.
        trading_days: Number of trading days per year.

    Returns:
        Annualised Sharpe ratio.
    """
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / trading_days
    std = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(trading_days))


def sortino_ratio(
    equity_series: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualised Sortino ratio.

    Args:
        equity_series: Time-indexed portfolio value series.
        risk_free: Daily risk-free rate.
        trading_days: Number of trading days per year.

    Returns:
        Annualised Sortino ratio.
    """
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / trading_days
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(trading_days))


def calmar_ratio(
    equity_series: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Calmar ratio (CAGR / Max Drawdown).

    Args:
        equity_series: Time-indexed portfolio value series.
        trading_days: Number of trading days per year.

    Returns:
        Calmar ratio.
    """
    annual = cagr(equity_series, trading_days)
    max_dd = max_drawdown(equity_series)
    if max_dd == 0:
        return float("inf") if annual > 0 else 0.0
    return annual / max_dd


def max_drawdown(equity_series: pd.Series) -> float:
    """Calculate maximum drawdown percentage.

    Args:
        equity_series: Time-indexed portfolio value series.

    Returns:
        Maximum drawdown as a positive percentage (e.g. 6.9 for 6.9%).
    """
    if len(equity_series) < 2:
        return 0.0
    running_max = equity_series.cummax()
    drawdowns = (running_max - equity_series) / running_max * 100.0
    return float(drawdowns.max())


def profit_factor(trades: list[dict[str, Any]]) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        trades: List of trade dictionaries with a ``pnl`` key.

    Returns:
        Profit factor. Returns ``inf`` if no losing trades.
    """
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy(trades: list[dict[str, Any]]) -> float:
    """Calculate trade expectancy (average PnL per trade).

    Args:
        trades: List of trade dictionaries with a ``pnl`` key.

    Returns:
        Expected value per trade.
    """
    if not trades:
        return 0.0
    return sum(t["pnl"] for t in trades) / len(trades)


def recovery_factor(equity_series: pd.Series) -> float:
    """Calculate recovery factor (total return / max drawdown).

    Args:
        equity_series: Time-indexed portfolio value series.

    Returns:
        Recovery factor.
    """
    tr = total_return(equity_series)
    max_dd = max_drawdown(equity_series)
    if max_dd == 0:
        return float("inf") if tr > 0 else 0.0
    return tr / max_dd


def win_rate(trades: list[dict[str, Any]]) -> float:
    """Calculate win rate percentage.

    Args:
        trades: List of trade dictionaries with a ``pnl`` key.

    Returns:
        Win rate as a percentage.
    """
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t["pnl"] > 0)
    return (winners / len(trades)) * 100.0


def loss_rate(trades: list[dict[str, Any]]) -> float:
    """Calculate loss rate percentage.

    Args:
        trades: List of trade dictionaries with a ``pnl`` key.

    Returns:
        Loss rate as a percentage.
    """
    return 100.0 - win_rate(trades)


def average_trade(trades: list[dict[str, Any]]) -> float:
    """Calculate average trade return percentage.

    Args:
        trades: List of trade dictionaries with a ``pnl_pct`` key.

    Returns:
        Average trade return percentage.
    """
    if not trades:
        return 0.0
    return sum(t.get("pnl_pct", 0.0) for t in trades) / len(trades) * 100.0


def average_holding_time(trades: list[dict[str, Any]]) -> float:
    """Calculate average holding time in bars.

    Args:
        trades: List of trade dictionaries with a ``bars_held`` key.

    Returns:
        Average number of bars held.
    """
    if not trades:
        return 0.0
    return sum(t.get("bars_held", 0) for t in trades) / len(trades)


def compute_all(
    equity_series: pd.Series,
    trades: list[dict[str, Any]],
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute all performance statistics at once.

    Args:
        equity_series: Time-indexed portfolio value series.
        trades: List of trade dictionaries.
        trading_days: Trading days per year.

    Returns:
        Dictionary of all computed metrics.
    """
    return {
        "total_return": round(total_return(equity_series), 2),
        "annual_return": round(annual_return(equity_series, trading_days), 2),
        "cagr": round(cagr(equity_series, trading_days), 2),
        "sharpe_ratio": round(sharpe_ratio(equity_series, trading_days=trading_days), 2),
        "sortino_ratio": round(sortino_ratio(equity_series, trading_days=trading_days), 2),
        "calmar_ratio": round(calmar_ratio(equity_series, trading_days), 2),
        "max_drawdown": round(max_drawdown(equity_series), 2),
        "profit_factor": round(profit_factor(trades), 2),
        "expectancy": round(expectancy(trades), 2),
        "recovery_factor": round(recovery_factor(equity_series), 2),
        "win_rate": round(win_rate(trades), 1),
        "loss_rate": round(loss_rate(trades), 1),
        "average_trade": round(average_trade(trades), 2),
        "average_holding_time": round(average_holding_time(trades), 1),
        "total_trades": len(trades),
    }
