"""Optimizer orchestrator.

Facade that wraps parameter search strategies and returns the best
parameter set with its performance metrics.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings
from backtesting.commission_model import CommissionModel
from backtesting.slippage_model import SlippageModel
from backtesting.parameter_search import (
    ParameterSpace,
    GridSearch,
    RandomSearch,
)


class Optimizer:
    """Strategy parameter optimiser.

    Orchestrates grid or random search over a user-defined parameter
    space and returns the best-performing configuration.

    Args:
        settings: Base application settings.
        df: OHLCV DataFrame for backtesting.
        commission_model: Optional commission model.
        slippage_model: Optional slippage model.
        initial_capital: Starting capital.
    """

    def __init__(
        self,
        settings: Settings,
        df: pd.DataFrame,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self._settings = settings
        self._df = df
        self._commission = commission_model
        self._slippage = slippage_model
        self._initial_capital = initial_capital
        self._best_params: dict[str, Any] = {}
        self._all_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        param_space: ParameterSpace,
        method: str = "grid",
        n_samples: int = 50,
        rank_by: str = "sharpe_ratio",
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run parameter optimisation.

        Args:
            param_space: Search space definition.
            method: ``'grid'`` or ``'random'``.
            n_samples: Samples for random search.
            rank_by: Metric to optimise.
            seed: Random seed.

        Returns:
            Dictionary with best parameters, metrics, and all results.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("  OPTIMIZER — {} Search", method.upper())
        logger.info("  Parameters: {} | Grid size: {}",
                     param_space.names, param_space.grid_size)
        logger.info("=" * 60)

        if method == "random":
            searcher = RandomSearch(
                settings=self._settings,
                param_space=param_space,
                df=self._df,
                n_samples=n_samples,
                seed=seed,
                commission_model=self._commission,
                slippage_model=self._slippage,
                initial_capital=self._initial_capital,
                rank_by=rank_by,
            )
        else:
            searcher = GridSearch(
                settings=self._settings,
                param_space=param_space,
                df=self._df,
                commission_model=self._commission,
                slippage_model=self._slippage,
                initial_capital=self._initial_capital,
                rank_by=rank_by,
            )

        self._all_results = searcher.run()
        self._best_params = searcher.best

        elapsed = time.time() - start

        logger.info("=" * 60)
        logger.info("  OPTIMIZATION COMPLETE — {:.2f}s", elapsed)
        if self._best_params:
            logger.info("  Best {}: {:.4f}",
                         rank_by, self._best_params.get(rank_by, 0))
            logger.info("  Best Params: {}", self._best_params.get("params", {}))
        logger.info("=" * 60)

        return {
            "best_params": self._best_params.get("params", {}),
            "best_metrics": {
                k: v for k, v in self._best_params.items() if k != "params"
            } if self._best_params else {},
            "all_results": self._all_results,
            "method": method,
            "total_evaluated": len(self._all_results),
            "execution_time": round(elapsed, 2),
        }

    @staticmethod
    def default_param_space() -> ParameterSpace:
        """Create a default parameter space for the HMM strategy.

        Returns:
            A :class:`ParameterSpace` with sensible defaults.
        """
        space = ParameterSpace()
        space.add("atr_multiplier_stop_loss", [1.5, 2.0, 2.5, 3.0])
        space.add("atr_multiplier_take_profit", [3.0, 4.0, 5.0, 6.0])
        space.add("risk_percentage", [0.5, 1.0, 1.5, 2.0])
        return space

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def best_params(self) -> dict[str, Any]:
        """Best parameter combination found."""
        return dict(self._best_params)

    @property
    def all_results(self) -> list[dict[str, Any]]:
        """All evaluated results."""
        return list(self._all_results)
