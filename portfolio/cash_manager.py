"""Cash Manager.

Handles safe locking, unlocking, and calculation of available buying power.
Wraps the Phase 5 Account object.
"""

from __future__ import annotations

from loguru import logger

from portfolio.account import Account


class CashManager:
    """Manages cash allocation and margin."""

    def __init__(self, account: Account) -> None:
        self.account = account

    def get_buying_power(self, leverage: float = 1.0) -> float:
        """Calculate total buying power considering leverage."""
        return self.account.cash * leverage

    def allocate_funds(self, required_cash: float) -> bool:
        """Lock cash for a new position."""
        if required_cash > self.account.cash:
            logger.warning(
                f"CashManager: Insufficient funds. Required: {required_cash:.2f}, Available: {self.account.cash:.2f}"
            )
            return False
            
        success = self.account.allocate_margin(required_cash)
        if success:
            logger.debug(f"CashManager: Allocated {required_cash:.2f} to margin. Cash remaining: {self.account.cash:.2f}")
        return success

    def release_funds(self, released_margin: float, realized_pnl: float) -> None:
        """Release margin back to cash and apply PnL."""
        self.account.release_margin(released_margin, realized_pnl)
        logger.debug(
            f"CashManager: Released {released_margin:.2f} margin with {realized_pnl:.2f} PnL. "
            f"New Cash: {self.account.cash:.2f}"
        )
