"""Walk-forward testing framework.

Implements rolling-window train/test splits with automatic model
retraining and per-window backtesting for out-of-sample validation.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings
from backtesting.backtest_engine import BacktestEngine
from backtesting.commission_model import CommissionModel
from backtesting.slippage_model import SlippageModel


class WalkForwardTester:
    """Rolling-window walk-forward backtester.

    Args:
        settings: Application settings.
        train_bars: Number of bars in each training window.
        test_bars: Number of bars in each testing window.
        step_bars: Number of bars to step forward between windows.
        commission_model: Optional commission model.
        slippage_model: Optional slippage model.
        initial_capital: Starting capital for each window.
    """

    def __init__(
        self,
        settings: Settings,
        train_bars: int = 252,
        test_bars: int = 63,
        step_bars: int = 63,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self._settings = settings
        self._train_bars = train_bars
        self._test_bars = test_bars
        self._step_bars = step_bars
        self._commission = commission_model or CommissionModel()
        self._slippage = slippage_model or SlippageModel()
        self._initial_capital = initial_capital
        self._window_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        *,
        retrain_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Execute walk-forward testing.

        Args:
            df: Full OHLCV DataFrame.
            retrain_fn: Optional callable ``(train_df) -> (strategy_manager, risk_manager, regime_detector)``
                that retrains the model on each training window.

        Returns:
            Aggregated walk-forward results.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("  WALK-FORWARD TEST — Starting")
        logger.info("  Train={} | Test={} | Step={} | Total Bars={}",
                     self._train_bars, self._test_bars, self._step_bars, len(df))
        logger.info("=" * 60)

        windows = self._create_windows(len(df))
        logger.info("Created {} walk-forward windows", len(windows))

        self._window_results = []

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_df = df.iloc[train_start:train_end].reset_index(drop=True)
            test_df = df.iloc[test_start:test_end].reset_index(drop=True)

            logger.info(
                "Window {}/{}: Train [{}-{}] Test [{}-{}]",
                idx + 1, len(windows),
                train_start, train_end, test_start, test_end,
            )

            strategy_mgr = None
            risk_mgr = None
            detector = None

            if retrain_fn is not None:
                try:
                    strategy_mgr, risk_mgr, detector = retrain_fn(train_df)
                    logger.info("Model retrained on window {}", idx + 1)
                except Exception as exc:
                    logger.warning("Retraining failed on window {}: {}", idx + 1, exc)

            engine = BacktestEngine(
                settings=self._settings,
                commission_model=self._commission,
                slippage_model=self._slippage,
                initial_capital=self._initial_capital,
            )

            result = engine.run(
                test_df,
                strategy_manager=strategy_mgr,
                risk_manager=risk_mgr,
                regime_detector=detector,
            )

            window_info = {
                "window": idx + 1,
                "train_range": (train_start, train_end),
                "test_range": (test_start, test_end),
                "total_return": result.get("total_return", 0.0),
                "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                "max_drawdown": result.get("max_drawdown", 0.0),
                "total_trades": result.get("total_trades", 0),
                "win_rate": result.get("win_rate", 0.0),
                "profit_factor": result.get("profit_factor", 0.0),
            }
            self._window_results.append(window_info)

        elapsed = time.time() - start_time
        return self._aggregate_results(elapsed)

    # ------------------------------------------------------------------
    # Window creation
    # ------------------------------------------------------------------

    def _create_windows(
        self, total_bars: int
    ) -> list[tuple[int, int, int, int]]:
        """Create train/test window index pairs."""
        windows: list[tuple[int, int, int, int]] = []
        start = 0

        while start + self._train_bars + self._test_bars <= total_bars:
            train_start = start
            train_end = start + self._train_bars
            test_start = train_end
            test_end = min(test_start + self._test_bars, total_bars)

            windows.append((train_start, train_end, test_start, test_end))
            start += self._step_bars

        return windows

    # ------------------------------------------------------------------
    # Results aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(self, elapsed: float) -> dict[str, Any]:
        """Aggregate results across all windows."""
        if not self._window_results:
            return {"windows": [], "aggregate": {}, "execution_time": elapsed}

        returns = [w["total_return"] for w in self._window_results]
        sharpes = [w["sharpe_ratio"] for w in self._window_results]
        drawdowns = [w["max_drawdown"] for w in self._window_results]
        trades = [w["total_trades"] for w in self._window_results]

        import numpy as np

        aggregate = {
            "total_windows": len(self._window_results),
            "avg_return": round(float(np.mean(returns)), 2),
            "median_return": round(float(np.median(returns)), 2),
            "std_return": round(float(np.std(returns)), 2),
            "avg_sharpe": round(float(np.mean(sharpes)), 2),
            "avg_max_drawdown": round(float(np.mean(drawdowns)), 2),
            "worst_drawdown": round(float(max(drawdowns)), 2),
            "total_trades_all_windows": sum(trades),
            "profitable_windows": sum(1 for r in returns if r > 0),
            "profitable_window_pct": round(
                sum(1 for r in returns if r > 0) / len(returns) * 100, 1
            ),
        }

        logger.info("Walk-forward aggregate: Avg Return={:.2f}%, Avg Sharpe={:.2f}",
                     aggregate["avg_return"], aggregate["avg_sharpe"])

        return {
            "windows": self._window_results,
            "aggregate": aggregate,
            "execution_time": round(elapsed, 2),
        }

    @property
    def window_results(self) -> list[dict[str, Any]]:
        """Results for each individual window."""
        return list(self._window_results)
