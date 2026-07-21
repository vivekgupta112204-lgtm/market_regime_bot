"""Position sizing logic.

Calculates the optimal order size based on risk parameters, ATR,
and account equity. Supports Fixed Fractional, ATR Risk, and a
simplified Kelly Criterion approach.
"""

from __future__ import annotations

from loguru import logger

from config.settings import StrategySettings


class PositionSizer:
    """Calculate position sizes based on risk constraints.

    Args:
        settings: Application strategy settings.
    """

    def __init__(self, settings: StrategySettings) -> None:
        self.settings = settings

    def calculate_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        method: str = "atr_risk",
        win_rate: float | None = None,
        risk_reward: float | None = None,
    ) -> float:
        """Calculate the number of units to trade.

        Args:
            capital: Current available capital.
            entry_price: Expected entry price.
            stop_loss: Expected stop loss price.
            method: 'fixed_fractional', 'atr_risk', or 'kelly'.
            win_rate: Historical win rate (required for Kelly).
            risk_reward: Historical risk/reward (required for Kelly).

        Returns:
            Number of units to trade, constrained by min/max limits.
        """
        if entry_price <= 0:
            return 0.0

        risk_amount = capital * (self.settings.risk_percentage / 100.0)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            logger.warning("Risk per unit is 0 (entry == stop), returning 0 size.")
            return 0.0

        # Method: Fixed Fractional
        if method == "fixed_fractional":
            raw_size = (capital * self.settings.leverage) / entry_price
            target_size = raw_size * (self.settings.risk_percentage / 100.0)

        # Method: Kelly Criterion
        elif method == "kelly" and win_rate is not None and risk_reward is not None:
            if risk_reward <= 0:
                kelly_pct = 0.0
            else:
                kelly_pct = win_rate - ((1 - win_rate) / risk_reward)
            # Use Half-Kelly for safety, capped by max risk.
            safe_kelly = max(0.0, min(kelly_pct / 2.0, self.settings.risk_percentage / 100.0))
            target_capital = capital * safe_kelly * self.settings.leverage
            target_size = target_capital / entry_price

        # Method: ATR Risk (Default)
        else:
            target_size = risk_amount / risk_per_unit * self.settings.leverage

        # Apply constraints.
        max_units = self.settings.max_position_size / entry_price
        final_size = min(target_size, max_units)

        if final_size < self.settings.min_lot_size:
            logger.debug(
                "Target size {:.4f} is below min lot size {:.4f}, skipping trade.",
                final_size,
                self.settings.min_lot_size,
            )
            return 0.0

        return round(final_size, 4)
