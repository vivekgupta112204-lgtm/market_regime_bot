"""Equity curve processing and derived analytics.

Processes daily equity snapshots into cumulative returns, rolling
returns, and rolling Sharpe ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


TRADING_DAYS_PER_YEAR = 252


class EquityCurve:
    """Process and analyse an equity time series.

    Args:
        equity_series: Time-indexed portfolio value series.
    """

    def __init__(self, equity_series: pd.Series) -> None:
        self._equity = equity_series.copy()
        self._returns: pd.Series | None = None

    @property
    def values(self) -> pd.Series:
        """Raw equity values."""
        return self._equity.copy()

    @property
    def returns(self) -> pd.Series:
        """Daily percentage returns."""
        if self._returns is None:
            self._returns = self._equity.pct_change().dropna()
        return self._returns.copy()

    @property
    def cumulative_returns(self) -> pd.Series:
        """Cumulative return series (starting at 0)."""
        if len(self._equity) < 2:
            return pd.Series(dtype=float)
        return (self._equity / self._equity.iloc[0] - 1.0) * 100.0

    @property
    def log_returns(self) -> pd.Series:
        """Daily log returns."""
        return np.log(self._equity / self._equity.shift(1)).dropna()

    def rolling_returns(self, window: int = 21) -> pd.Series:
        """Calculate rolling returns over a specified window.

        Args:
            window: Number of bars in the rolling window.

        Returns:
            Rolling return series as a percentage.
        """
        if len(self._equity) < window:
            return pd.Series(dtype=float)
        return (self._equity / self._equity.shift(window) - 1.0).dropna() * 100.0

    def rolling_sharpe(
        self,
        window: int = 63,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio.

        Args:
            window: Number of bars in the rolling window.
            trading_days: Trading days per year for annualisation.

        Returns:
            Rolling Sharpe ratio series.
        """
        rets = self.returns
        if len(rets) < window:
            return pd.Series(dtype=float)
        rolling_mean = rets.rolling(window).mean()
        rolling_std = rets.rolling(window).std()
        rolling_std = rolling_std.replace(0, np.nan)
        return (rolling_mean / rolling_std * np.sqrt(trading_days)).dropna()

    def rolling_volatility(
        self,
        window: int = 21,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """Calculate rolling annualised volatility.

        Args:
            window: Number of bars in the rolling window.
            trading_days: Trading days per year for annualisation.

        Returns:
            Rolling volatility series as a percentage.
        """
        rets = self.returns
        if len(rets) < window:
            return pd.Series(dtype=float)
        return (rets.rolling(window).std() * np.sqrt(trading_days) * 100.0).dropna()

    def monthly_returns(self) -> pd.DataFrame:
        """Compute monthly return table.

        Returns:
            DataFrame with years as rows, months as columns, and
            percentage returns as values.
        """
        if len(self._equity) < 2:
            return pd.DataFrame()

        equity = self._equity.copy()
        if not isinstance(equity.index, pd.DatetimeIndex):
            try:
                equity.index = pd.to_datetime(equity.index)
            except Exception:
                return pd.DataFrame()

        monthly = equity.resample("ME").last()
        monthly_ret = monthly.pct_change().dropna() * 100.0
        monthly_ret.index = pd.to_datetime(monthly_ret.index)

        table = pd.DataFrame(index=sorted(monthly_ret.index.year.unique()))
        for month in range(1, 13):
            mask = monthly_ret.index.month == month
            subset = monthly_ret[mask]
            table[month] = subset.values[:len(table)] if len(subset) > 0 else np.nan

        table.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        return table

    def yearly_returns(self) -> pd.Series:
        """Compute yearly returns.

        Returns:
            Series of annual returns as percentages.
        """
        if len(self._equity) < 2:
            return pd.Series(dtype=float)

        equity = self._equity.copy()
        if not isinstance(equity.index, pd.DatetimeIndex):
            try:
                equity.index = pd.to_datetime(equity.index)
            except Exception:
                return pd.Series(dtype=float)

        yearly = equity.resample("YE").last()
        return (yearly.pct_change().dropna() * 100.0)
