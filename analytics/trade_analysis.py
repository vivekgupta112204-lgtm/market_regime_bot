"""Trade-level analytics for backtesting results.

Analyses winning/losing trades, streaks, duration distribution,
best/worst trade, and PnL distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TradeStats:
    """Aggregated statistics for a set of trades.

    Attributes:
        count: Number of trades.
        total_pnl: Sum of PnL.
        average_pnl: Mean PnL.
        median_pnl: Median PnL.
        std_pnl: Standard deviation of PnL.
        average_pnl_pct: Mean return percentage.
        average_duration: Mean holding time in bars.
    """

    count: int = 0
    total_pnl: float = 0.0
    average_pnl: float = 0.0
    median_pnl: float = 0.0
    std_pnl: float = 0.0
    average_pnl_pct: float = 0.0
    average_duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "count": self.count,
            "total_pnl": round(self.total_pnl, 2),
            "average_pnl": round(self.average_pnl, 2),
            "median_pnl": round(self.median_pnl, 2),
            "std_pnl": round(self.std_pnl, 2),
            "average_pnl_pct": round(self.average_pnl_pct, 4),
            "average_duration": round(self.average_duration, 1),
        }


def _compute_stats(trades: list[dict[str, Any]]) -> TradeStats:
    """Compute aggregate statistics for a list of trade dicts."""
    if not trades:
        return TradeStats()

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t.get("pnl_pct", 0.0) for t in trades]
    durations = [t.get("bars_held", 0) for t in trades]

    return TradeStats(
        count=len(trades),
        total_pnl=sum(pnls),
        average_pnl=float(np.mean(pnls)),
        median_pnl=float(np.median(pnls)),
        std_pnl=float(np.std(pnls)) if len(pnls) > 1 else 0.0,
        average_pnl_pct=float(np.mean(pnl_pcts)),
        average_duration=float(np.mean(durations)),
    )


def winning_trades(trades: list[dict[str, Any]]) -> TradeStats:
    """Compute statistics for winning trades.

    Args:
        trades: List of trade dictionaries with ``pnl`` key.

    Returns:
        :class:`TradeStats` for trades with positive PnL.
    """
    winners = [t for t in trades if t["pnl"] > 0]
    return _compute_stats(winners)


def losing_trades(trades: list[dict[str, Any]]) -> TradeStats:
    """Compute statistics for losing trades.

    Args:
        trades: List of trade dictionaries with ``pnl`` key.

    Returns:
        :class:`TradeStats` for trades with negative PnL.
    """
    losers = [t for t in trades if t["pnl"] < 0]
    return _compute_stats(losers)


def best_trade(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the trade with the highest PnL.

    Args:
        trades: List of trade dictionaries.

    Returns:
        The best-performing trade dictionary, or empty dict.
    """
    if not trades:
        return {}
    return max(trades, key=lambda t: t["pnl"])


def worst_trade(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the trade with the lowest PnL.

    Args:
        trades: List of trade dictionaries.

    Returns:
        The worst-performing trade dictionary, or empty dict.
    """
    if not trades:
        return {}
    return min(trades, key=lambda t: t["pnl"])


def longest_win_streak(trades: list[dict[str, Any]]) -> int:
    """Calculate the longest consecutive winning streak.

    Args:
        trades: List of trade dictionaries with ``pnl`` key.

    Returns:
        Length of longest win streak.
    """
    return _longest_streak(trades, winning=True)


def longest_loss_streak(trades: list[dict[str, Any]]) -> int:
    """Calculate the longest consecutive losing streak.

    Args:
        trades: List of trade dictionaries with ``pnl`` key.

    Returns:
        Length of longest loss streak.
    """
    return _longest_streak(trades, winning=False)


def _longest_streak(trades: list[dict[str, Any]], winning: bool) -> int:
    """Generic streak calculator."""
    if not trades:
        return 0
    max_streak = 0
    current = 0
    for t in trades:
        if (t["pnl"] > 0) == winning:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def trade_duration_distribution(trades: list[dict[str, Any]]) -> dict[str, int]:
    """Bucket trades by holding duration.

    Args:
        trades: List of trade dictionaries with ``bars_held`` key.

    Returns:
        Dictionary mapping duration buckets to trade counts.
    """
    if not trades:
        return {}

    buckets = {
        "1_bar": 0,
        "2_5_bars": 0,
        "6_10_bars": 0,
        "11_20_bars": 0,
        "21_50_bars": 0,
        "50_plus_bars": 0,
    }

    for t in trades:
        bars = t.get("bars_held", 0)
        if bars <= 1:
            buckets["1_bar"] += 1
        elif bars <= 5:
            buckets["2_5_bars"] += 1
        elif bars <= 10:
            buckets["6_10_bars"] += 1
        elif bars <= 20:
            buckets["11_20_bars"] += 1
        elif bars <= 50:
            buckets["21_50_bars"] += 1
        else:
            buckets["50_plus_bars"] += 1

    return buckets


def pnl_distribution(trades: list[dict[str, Any]], bins: int = 20) -> dict[str, Any]:
    """Compute PnL distribution histogram data.

    Args:
        trades: List of trade dictionaries with ``pnl`` key.
        bins: Number of histogram bins.

    Returns:
        Dictionary with ``bin_edges`` and ``counts``.
    """
    if not trades:
        return {"bin_edges": [], "counts": []}

    pnls = [t["pnl"] for t in trades]
    counts, edges = np.histogram(pnls, bins=bins)
    return {
        "bin_edges": [round(float(e), 2) for e in edges],
        "counts": [int(c) for c in counts],
    }


def compute_all(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all trade-level analytics.

    Args:
        trades: List of trade dictionaries.

    Returns:
        Comprehensive trade analysis dictionary.
    """
    bt = best_trade(trades)
    wt = worst_trade(trades)

    return {
        "total_trades": len(trades),
        "winning": winning_trades(trades).to_dict(),
        "losing": losing_trades(trades).to_dict(),
        "best_trade_pnl": round(bt.get("pnl", 0.0), 2) if bt else 0.0,
        "best_trade_pct": round(bt.get("pnl_pct", 0.0) * 100, 2) if bt else 0.0,
        "worst_trade_pnl": round(wt.get("pnl", 0.0), 2) if wt else 0.0,
        "worst_trade_pct": round(wt.get("pnl_pct", 0.0) * 100, 2) if wt else 0.0,
        "longest_win_streak": longest_win_streak(trades),
        "longest_loss_streak": longest_loss_streak(trades),
        "duration_distribution": trade_duration_distribution(trades),
        "pnl_distribution": pnl_distribution(trades),
    }
