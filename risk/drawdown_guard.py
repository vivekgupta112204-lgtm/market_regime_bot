"""Drawdown Guard.

Monitors portfolio high-water marks and rejects trades if drawdown limits are breached.
"""

from __future__ import annotations

from loguru import logger

from config.settings import RiskSettings
from portfolio.portfolio_state import PortfolioState


class DrawdownGuard:
    """Evaluates the portfolio's current drawdown against the maximum allowed limit.

    Args:
        settings: Risk settings containing max_drawdown_pct.
    """

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    def is_safe(self, portfolio: PortfolioState) -> bool:
        """Check if current drawdown is within safe limits.

        Args:
            portfolio: The current state of the portfolio.

        Returns:
            True if trading is allowed (drawdown < limit), False otherwise.
        """
        current_equity = portfolio.total_equity
        drawdown_pct = portfolio.account.get_drawdown(current_equity)

        if drawdown_pct > self.settings.max_drawdown_pct:
            logger.warning(
                "Drawdown Guard Triggered: Current DD ({:.2%}) exceeds limit ({:.2%}).",
                drawdown_pct,
                self.settings.max_drawdown_pct,
            )
            return False
            
        return True
