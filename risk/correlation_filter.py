"""Correlation Filter.

Simplified correlation tracking to prevent heavy concentration risk.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from config.settings import RiskSettings
from portfolio.portfolio_state import PortfolioState


class CorrelationFilter:
    """Evaluates whether adding a new asset exceeds correlation limits.

    In a full production setup, this would maintain a rolling DataFrame
    of returns for all traded assets and compute Pearson correlation.
    For this engine, we mock the correlation check assuming an external
    oracle or simplified sector grouping.
    """

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    def _mock_correlation(self, sym1: str, sym2: str) -> float:
        """Mock correlation function. In reality, pull from historical data."""
        # For demo purposes, we assume identical symbols have 1.0 correlation,
        # and different symbols have 0.3 (safe).
        return 1.0 if sym1 == sym2 else 0.3

    def is_safe(self, target_symbol: str, portfolio: PortfolioState) -> bool:
        """Check if trading the target symbol violates correlation bounds.

        Args:
            target_symbol: The asset attempting to be traded.
            portfolio: The current portfolio state.

        Returns:
            True if correlation is below the threshold, False otherwise.
        """
        if not portfolio.positions:
            return True

        # Check correlation against all open positions
        for open_sym in portfolio.positions.keys():
            corr = self._mock_correlation(target_symbol, open_sym)
            if corr >= self.settings.correlation_threshold:
                logger.warning(
                    "Correlation Filter Reject: {} is highly correlated ({:.2f}) with open position {}",
                    target_symbol,
                    corr,
                    open_sym,
                )
                return False

        return True
