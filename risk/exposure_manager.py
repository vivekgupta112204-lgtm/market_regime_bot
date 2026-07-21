"""Exposure Manager.

Enforces limits on total portfolio exposure, single asset exposure,
and maximum concurrent open positions.
"""

from __future__ import annotations

from loguru import logger

from config.settings import RiskSettings
from portfolio.portfolio_state import PortfolioState


class ExposureManager:
    """Validates trade sizing against exposure constraints."""

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    def is_safe(
        self,
        target_symbol: str,
        target_fiat_size: float,
        portfolio: PortfolioState,
    ) -> bool:
        """Check if adding the requested position violates exposure limits.

        Args:
            target_symbol: Asset being traded.
            target_fiat_size: Proposed position size in fiat currency.
            portfolio: Current portfolio state.

        Returns:
            True if safe, False if limits exceeded.
        """
        # 1. Max Open Positions Limit
        if target_symbol not in portfolio.positions:
            if len(portfolio.positions) >= self.settings.max_open_positions:
                logger.warning(
                    "Exposure Reject: Max open positions reached ({} >= {}).",
                    len(portfolio.positions),
                    self.settings.max_open_positions,
                )
                return False

        # 2. Total Portfolio Exposure Limit
        # Calculate new total exposure
        current_exposure = portfolio.total_exposure
        projected_exposure = current_exposure + target_fiat_size
        max_allowed_exposure = portfolio.account.total_equity * self.settings.max_portfolio_exposure_pct

        if projected_exposure > max_allowed_exposure:
            logger.warning(
                "Exposure Reject: Target exposure ({:.2f}) exceeds max allowed ({:.2f}).",
                projected_exposure,
                max_allowed_exposure,
            )
            return False

        # 3. Available Margin Check
        # Note: If trading on margin/leverage, 'cash' is the bottleneck.
        # Assuming 1:1 required margin for this check (handled precisely in LeverageManager).
        if target_fiat_size > portfolio.account.cash:
            logger.warning(
                "Exposure Reject: Insufficient cash ({:.2f} required, {:.2f} available).",
                target_fiat_size,
                portfolio.account.cash,
            )
            return False

        return True
