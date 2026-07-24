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
        k_folds: int = 5,
        embargo_bars: int = 50,
        num_trials: int = 1,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self._settings = settings
        self._k_folds = k_folds
        self._embargo_bars = embargo_bars
        self._num_trials = num_trials # For Deflated Sharpe Ratio penalty
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
        logger.info("  CPCV TEST — Purged K-Fold Cross-Validation Starting")
        logger.info("  Folds={} | Embargo={} | Total Bars={}",
                     self._k_folds, self._embargo_bars, len(df))
        logger.info("=" * 60)

        windows = self._create_windows(len(df))
        logger.info("Created {} CPCV purged folds", len(windows))

        self._window_results = []

        for idx, (train_indices, test_start, test_end) in enumerate(windows):
            # train_df might be disjointed, so we concat the valid subsets
            train_df = df.iloc[train_indices].reset_index(drop=True)
            test_df = df.iloc[test_start:test_end].reset_index(drop=True)

            logger.info(
                "Fold {}/{}: Test [{}-{}] | Train Size: {}",
                idx + 1, len(windows),
                test_start, test_end, len(train_df)
            )

            strategy_mgr = None
            risk_mgr = None
            detector = None

            if retrain_fn is not None:
                try:
                    strategy_mgr, risk_mgr, detector = retrain_fn(train_df)
                    logger.info("Model retrained on purged fold {}", idx + 1)
                except Exception as exc:
                    logger.warning("Retraining failed on fold {}: {}", idx + 1, exc)

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
    # CPCV Purging & Embargoing
    # ------------------------------------------------------------------

    def _create_windows(
        self, total_bars: int
    ) -> list[tuple[list[int], int, int]]:
        """Create test folds and purged/embargoed train indices."""
        import numpy as np
        windows = []
        fold_size = total_bars // self._k_folds
        
        all_indices = np.arange(total_bars)
        
        for k in range(self._k_folds):
            test_start = k * fold_size
            test_end = test_start + fold_size if k < self._k_folds - 1 else total_bars
            
            # Apply Purge (before) and Embargo (after)
            purge_start = max(0, test_start - self._embargo_bars)
            embargo_end = min(total_bars, test_end + self._embargo_bars)
            
            # Train indices are everything outside [purge_start, embargo_end]
            train_mask = np.ones(total_bars, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            
            train_indices = all_indices[train_mask].tolist()
            windows.append((train_indices, test_start, test_end))

        return windows

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

        avg_sharpe = float(np.mean(sharpes))
        # Deflated Sharpe Ratio Penalty
        # Discount the Sharpe based on the theoretical expected Max Sharpe of random trials
        import math
        euler_mascheroni = 0.57721
        expected_max_sr = math.sqrt(2 * math.log(max(1, self._num_trials)))
        deflated_sharpe = avg_sharpe - expected_max_sr if self._num_trials > 1 else avg_sharpe

        aggregate = {
            "total_windows": len(self._window_results),
            "avg_return": round(float(np.mean(returns)), 2),
            "median_return": round(float(np.median(returns)), 2),
            "std_return": round(float(np.std(returns)), 2),
            "avg_sharpe": round(avg_sharpe, 2),
            "deflated_sharpe": round(float(deflated_sharpe), 2),
            "avg_max_drawdown": round(float(np.mean(drawdowns)), 2),
            "worst_drawdown": round(float(max(drawdowns)), 2),
            "total_trades_all_windows": sum(trades),
            "profitable_windows": sum(1 for r in returns if r > 0),
            "profitable_window_pct": round(
                sum(1 for r in returns if r > 0) / len(returns) * 100, 1
            ),
        }

        logger.info("CPCV Aggregate: Avg Return={:.2f}%, Avg Sharpe={:.2f} (Deflated={:.2f})",
                     aggregate["avg_return"], aggregate["avg_sharpe"], aggregate["deflated_sharpe"])

        return {
            "windows": self._window_results,
            "aggregate": aggregate,
            "execution_time": round(elapsed, 2),
        }

    @property
    def window_results(self) -> list[dict[str, Any]]:
        """Results for each individual window."""
        return list(self._window_results)
