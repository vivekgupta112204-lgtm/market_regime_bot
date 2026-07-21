"""Daily Loss Guard.

Tracks daily realized and unrealized PnL. Disables trading if daily limits are hit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from loguru import logger

from config.settings import RiskSettings
from portfolio.portfolio_state import PortfolioState


class DailyLossGuard:
    """Evaluates daily PnL and consecutive losses to halt trading if necessary.

    Args:
        settings: Risk settings containing max_daily_loss_fiat.
    """

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings
        self._current_date_str = ""
        self._daily_realized_pnl = 0.0
        self._consecutive_losers = 0

    def _check_rollover(self, timestamp: datetime) -> None:
        """Reset daily tracking counters if a new day has started."""
        date_str = timestamp.strftime("%Y-%m-%d")
        if date_str != self._current_date_str:
            self._current_date_str = date_str
            self._daily_realized_pnl = 0.0
            self._consecutive_losers = 0

    def is_safe(self, portfolio: PortfolioState, timestamp: datetime | None = None) -> bool:
        """Check if trading is allowed under daily loss rules.

        Args:
            portfolio: Current portfolio state.
            timestamp: Optional current timestamp (uses UTC now if None).

        Returns:
            True if safe, False if daily loss limits are exceeded.
        """
        ts = timestamp or datetime.now(timezone.utc)
        self._check_rollover(ts)

        # Include unrealized PnL in the daily loss calculation
        unrealized_pnl = sum(p.unrealized_pnl for p in portfolio.positions.values())
        total_daily_pnl = self._daily_realized_pnl + unrealized_pnl

        # Note: We check if the total daily PnL is less than the negative maximum loss
        if total_daily_pnl < -abs(self.settings.max_daily_loss_fiat):
            logger.warning(
                "Daily Loss Guard Triggered: Total Daily PnL ({:.2f}) exceeds max loss limit ({:.2f}).",
                total_daily_pnl,
                -abs(self.settings.max_daily_loss_fiat),
            )
            return False

        # Additional guard (could be parameterized)
        if self._consecutive_losers >= 5:
            logger.warning("Daily Loss Guard Triggered: 5 consecutive losing trades hit.")
            return False

        return True

    def record_closed_trade(self, pnl: float, timestamp: datetime | None = None) -> None:
        """Record the outcome of a closed trade."""
        ts = timestamp or datetime.now(timezone.utc)
        self._check_rollover(ts)
        
        self._daily_realized_pnl += pnl
        if pnl < 0:
            self._consecutive_losers += 1
        else:
            self._consecutive_losers = 0
