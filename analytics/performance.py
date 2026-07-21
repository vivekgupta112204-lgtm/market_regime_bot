"""Unified performance analytics facade.

Orchestrates all analytics sub-modules to produce a comprehensive
performance report from equity curve and trade data.  Supports
benchmark comparison against Buy & Hold, SPY, NASDAQ, or custom series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from analytics import statistics as stats_mod
from analytics import risk_metrics as risk_mod
from analytics.equity_curve import EquityCurve
from analytics.drawdown import DrawdownAnalyzer
from analytics import trade_analysis as trade_mod


class PerformanceAnalyzer:
    """Unified facade for computing all performance analytics.

    Args:
        equity_series: Time-indexed portfolio value series.
        trades: List of closed-trade dictionaries.
        benchmark_series: Optional benchmark equity series.
        initial_capital: Starting capital (for Buy & Hold comparison).
    """

    def __init__(
        self,
        equity_series: pd.Series,
        trades: list[dict[str, Any]],
        benchmark_series: pd.Series | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self._equity = equity_series
        self._trades = trades
        self._benchmark = benchmark_series
        self._initial_capital = initial_capital

        self._curve = EquityCurve(equity_series)
        self._drawdown = DrawdownAnalyzer(equity_series)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self) -> dict[str, Any]:
        """Run every analytics module and return a comprehensive report.

        Returns:
            Nested dictionary with all metrics, analytics, and comparisons.
        """
        logger.info("Computing comprehensive performance analytics...")

        perf = stats_mod.compute_all(self._equity, self._trades)
        risk = risk_mod.compute_all(self._equity, self._benchmark)
        dd = self._drawdown.compute_all()
        trades = trade_mod.compute_all(self._trades)

        result: dict[str, Any] = {
            "performance": perf,
            "risk": risk,
            "drawdown": dd,
            "trades": trades,
        }

        if self._benchmark is not None and len(self._benchmark) > 1:
            result["benchmark_comparison"] = self._compare_benchmarks()

        buy_hold = self._buy_and_hold_comparison()
        if buy_hold:
            result["buy_and_hold"] = buy_hold

        logger.info("Analytics complete: {} total trades, Sharpe={:.2f}",
                     perf.get("total_trades", 0), perf.get("sharpe_ratio", 0))

        return result

    def summary(self) -> dict[str, Any]:
        """Return a condensed performance summary.

        Returns:
            Flat dictionary with the most important metrics.
        """
        perf = stats_mod.compute_all(self._equity, self._trades)
        bt = trade_mod.best_trade(self._trades)
        wt = trade_mod.worst_trade(self._trades)

        return {
            "initial_capital": self._initial_capital,
            "final_capital": round(float(self._equity.iloc[-1]), 2) if len(self._equity) > 0 else self._initial_capital,
            "total_return": perf["total_return"],
            "annual_return": perf["annual_return"],
            "cagr": perf["cagr"],
            "sharpe_ratio": perf["sharpe_ratio"],
            "sortino_ratio": perf["sortino_ratio"],
            "max_drawdown": perf["max_drawdown"],
            "profit_factor": perf["profit_factor"],
            "win_rate": perf["win_rate"],
            "total_trades": perf["total_trades"],
            "average_trade": perf["average_trade"],
            "best_trade": round(bt.get("pnl_pct", 0.0) * 100, 2) if bt else 0.0,
            "worst_trade": round(wt.get("pnl_pct", 0.0) * 100, 2) if wt else 0.0,
        }

    # ------------------------------------------------------------------
    # Benchmark comparisons
    # ------------------------------------------------------------------

    def _compare_benchmarks(self) -> dict[str, Any]:
        """Compare strategy to the provided benchmark."""
        if self._benchmark is None:
            return {}

        bench_perf = stats_mod.compute_all(self._benchmark, [])
        strategy_perf = stats_mod.compute_all(self._equity, self._trades)

        return {
            "strategy": {
                "total_return": strategy_perf["total_return"],
                "sharpe_ratio": strategy_perf["sharpe_ratio"],
                "max_drawdown": strategy_perf["max_drawdown"],
            },
            "benchmark": {
                "total_return": bench_perf["total_return"],
                "sharpe_ratio": bench_perf["sharpe_ratio"],
                "max_drawdown": bench_perf["max_drawdown"],
            },
            "excess_return": round(
                strategy_perf["total_return"] - bench_perf["total_return"], 2
            ),
            "beta": risk_mod.beta(self._equity, self._benchmark),
            "alpha": risk_mod.alpha(self._equity, self._benchmark),
            "information_ratio": risk_mod.information_ratio(
                self._equity, self._benchmark
            ),
        }

    def _buy_and_hold_comparison(self) -> dict[str, Any]:
        """Compare strategy to a passive buy-and-hold approach.

        Uses first and last prices in the equity series to simulate.
        """
        if len(self._equity) < 2:
            return {}

        first_val = self._equity.iloc[0]
        last_val = self._equity.iloc[-1]

        bh_return = ((last_val / first_val) - 1.0) * 100.0 if first_val > 0 else 0.0
        strategy_return = stats_mod.total_return(self._equity)

        return {
            "buy_hold_return": round(bh_return, 2),
            "strategy_return": round(strategy_return, 2),
            "outperformance": round(strategy_return - bh_return, 2),
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def equity_curve(self) -> EquityCurve:
        """The underlying :class:`EquityCurve` object."""
        return self._curve

    @property
    def drawdown_analyzer(self) -> DrawdownAnalyzer:
        """The underlying :class:`DrawdownAnalyzer` object."""
        return self._drawdown
