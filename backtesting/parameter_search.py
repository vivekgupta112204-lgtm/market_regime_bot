"""Parameter search strategies for strategy optimisation.

Supports grid search and random search over a configurable parameter
space, running a backtest for each combination and ranking results.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings
from backtesting.backtest_engine import BacktestEngine
from backtesting.commission_model import CommissionModel
from backtesting.slippage_model import SlippageModel


class ParameterSpace:
    """Define a multi-dimensional parameter search space.

    Each parameter is a named dimension with a list of candidate values.

    Example::

        space = ParameterSpace()
        space.add("atr_multiplier_stop_loss", [1.5, 2.0, 2.5, 3.0])
        space.add("rsi_oversold", [25, 30, 35])
    """

    def __init__(self) -> None:
        self._params: dict[str, list[Any]] = {}

    def add(self, name: str, values: list[Any]) -> None:
        """Add a parameter dimension.

        Args:
            name: Parameter name.
            values: List of candidate values.
        """
        self._params[name] = list(values)

    @property
    def names(self) -> list[str]:
        """Parameter names."""
        return list(self._params.keys())

    @property
    def grid_size(self) -> int:
        """Total number of combinations in a full grid search."""
        if not self._params:
            return 0
        size = 1
        for v in self._params.values():
            size *= len(v)
        return size

    def grid_combinations(self) -> list[dict[str, Any]]:
        """Return all parameter combinations for grid search.

        Returns:
            List of parameter dictionaries.
        """
        if not self._params:
            return [{}]
        names = list(self._params.keys())
        values = list(self._params.values())
        return [dict(zip(names, combo)) for combo in itertools.product(*values)]

    def random_combinations(self, n: int, seed: int = 42) -> list[dict[str, Any]]:
        """Return N random parameter combinations.

        Args:
            n: Number of random combinations.
            seed: Random seed.

        Returns:
            List of parameter dictionaries.
        """
        rng = random.Random(seed)
        combos: list[dict[str, Any]] = []
        names = list(self._params.keys())
        values = list(self._params.values())

        for _ in range(n):
            combo = {name: rng.choice(vals) for name, vals in zip(names, values)}
            combos.append(combo)

        return combos


def _apply_params(settings: Settings, params: dict[str, Any]) -> Settings:
    """Apply parameter overrides to a Settings copy.

    Modifies strategy-related settings based on parameter names.

    Args:
        settings: Base settings.
        params: Parameter overrides.

    Returns:
        Modified settings (shallow copy of sub-models).
    """
    import copy
    s = settings.model_copy(deep=True)

    param_map = {
        "atr_multiplier_stop_loss": ("strategy", "atr_multiplier_stop_loss"),
        "atr_multiplier_take_profit": ("strategy", "atr_multiplier_take_profit"),
        "risk_percentage": ("strategy", "risk_percentage"),
        "min_regime_confidence": ("strategy", "min_regime_confidence"),
        "max_trades_per_day": ("strategy", "max_trades_per_day"),
        "leverage": ("strategy", "leverage"),
        "max_drawdown_pct": ("risk", "max_drawdown_pct"),
        "max_portfolio_exposure_pct": ("risk", "max_portfolio_exposure_pct"),
        "correlation_threshold": ("risk", "correlation_threshold"),
        "trailing_stop_activation_rr": ("risk", "trailing_stop_activation_rr"),
        "trailing_stop_distance_atr": ("risk", "trailing_stop_distance_atr"),
    }

    for name, value in params.items():
        if name in param_map:
            group, attr = param_map[name]
            sub = getattr(s, group)
            setattr(sub, attr, value)
        elif hasattr(s, name):
            setattr(s, name, value)

    return s


class GridSearch:
    """Exhaustive grid search over a parameter space.

    Args:
        settings: Base application settings.
        param_space: Parameter search space.
        df: OHLCV DataFrame for backtesting.
        commission_model: Optional commission model.
        slippage_model: Optional slippage model.
        initial_capital: Starting capital.
        rank_by: Metric to rank results (default: ``sharpe_ratio``).
    """

    def __init__(
        self,
        settings: Settings,
        param_space: ParameterSpace,
        df: pd.DataFrame,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        initial_capital: float = 100_000.0,
        rank_by: str = "sharpe_ratio",
    ) -> None:
        self._settings = settings
        self._space = param_space
        self._df = df
        self._commission = commission_model or CommissionModel()
        self._slippage = slippage_model or SlippageModel()
        self._initial_capital = initial_capital
        self._rank_by = rank_by
        self._results: list[dict[str, Any]] = []

    def run(self) -> list[dict[str, Any]]:
        """Execute grid search.

        Returns:
            Sorted list of results (best first).
        """
        combos = self._space.grid_combinations()
        logger.info("Grid search: {} combinations to evaluate", len(combos))
        return self._evaluate(combos)

    def _evaluate(self, combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Evaluate all parameter combinations."""
        self._results = []
        total = len(combos)

        for idx, params in enumerate(combos):
            start = time.time()
            modified_settings = _apply_params(self._settings, params)

            engine = BacktestEngine(
                settings=modified_settings,
                commission_model=self._commission,
                slippage_model=self._slippage,
                initial_capital=self._initial_capital,
            )

            try:
                result = engine.run(self._df)
                elapsed = time.time() - start

                entry = {
                    "params": params,
                    "total_return": result.get("total_return", 0.0),
                    "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                    "sortino_ratio": result.get("sortino_ratio", 0.0),
                    "max_drawdown": result.get("max_drawdown", 0.0),
                    "profit_factor": result.get("profit_factor", 0.0),
                    "win_rate": result.get("win_rate", 0.0),
                    "total_trades": result.get("total_trades", 0),
                    "calmar_ratio": result.get("analytics", {}).get("performance", {}).get("calmar_ratio", 0.0),
                    "execution_time": round(elapsed, 2),
                }
                self._results.append(entry)

                if (idx + 1) % 10 == 0 or idx == total - 1:
                    logger.info(
                        "Grid search: {}/{} complete | Best Sharpe so far: {:.2f}",
                        idx + 1, total,
                        max((r[self._rank_by] for r in self._results), default=0),
                    )

            except Exception as exc:
                logger.warning("Combination {} failed: {}", params, exc)

        self._results.sort(key=lambda r: r.get(self._rank_by, 0), reverse=True)
        return self._results

    @property
    def results(self) -> list[dict[str, Any]]:
        """All evaluated results, sorted by rank metric."""
        return list(self._results)

    @property
    def best(self) -> dict[str, Any]:
        """Best parameter combination."""
        return self._results[0] if self._results else {}


class RandomSearch(GridSearch):
    """Random search over a parameter space.

    Args:
        n_samples: Number of random combinations to evaluate.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        settings: Settings,
        param_space: ParameterSpace,
        df: pd.DataFrame,
        n_samples: int = 50,
        seed: int = 42,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        initial_capital: float = 100_000.0,
        rank_by: str = "sharpe_ratio",
    ) -> None:
        super().__init__(
            settings, param_space, df,
            commission_model, slippage_model,
            initial_capital, rank_by,
        )
        self._n_samples = n_samples
        self._seed = seed

    def run(self) -> list[dict[str, Any]]:
        """Execute random search.

        Returns:
            Sorted list of results (best first).
        """
        combos = self._space.random_combinations(self._n_samples, self._seed)
        logger.info("Random search: {} samples to evaluate", len(combos))
        return self._evaluate(combos)
