"""Advanced Position Sizing Engine.

Supports multiple models: Fixed Size, Fixed Fractional, % Risk, ATR Risk,
Volatility Adjusted, Kelly Criterion, Max Dollar Risk, and Max Portfolio Risk.
"""

from __future__ import annotations

import math
from loguru import logger

from config.settings import StrategySettings, RiskSettings
from portfolio.portfolio_state import PortfolioState


class AdvancedPositionSizer:
    """Calculates position sizes using multiple models.

    Args:
        strategy_settings: Base strategy settings.
        risk_settings: Portfolio-wide risk settings.
    """

    def __init__(self, strategy_settings: StrategySettings, risk_settings: RiskSettings) -> None:
        self.strategy_settings = strategy_settings
        self.risk_settings = risk_settings

    def calculate_size(
        self,
        portfolio: PortfolioState,
        entry_price: float,
        stop_loss: float,
        method: str = "atr_risk",
        fixed_size_units: float = 1.0,
        win_rate: float = 0.5,
        risk_reward: float = 2.0,
    ) -> float:
        """Calculate the number of units to trade.

        Args:
            portfolio: Current portfolio state.
            entry_price: Planned entry price.
            stop_loss: Planned stop loss price.
            method: Name of the sizing method.
            fixed_size_units: Target units if using 'fixed_size'.
            win_rate: Historical win rate (for Kelly).
            risk_reward: Historical risk/reward (for Kelly).

        Returns:
            Number of units to trade, bounded by portfolio limits.
        """
        if entry_price <= 0:
            return 0.0

        capital = portfolio.account.total_equity
        if capital <= 0:
            return 0.0

        risk_amount = capital * (self.strategy_settings.risk_percentage / 100.0)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0 and method in ("atr_risk", "percentage_risk", "max_dollar_risk"):
            logger.warning("Risk per unit is 0. Sizer returning 0.")
            return 0.0

        target_size = 0.0

        if method == "fixed_size":
            target_size = fixed_size_units
            
        elif method == "fixed_fractional":
            raw_size = (capital * self.strategy_settings.leverage) / entry_price
            target_size = raw_size * (self.strategy_settings.risk_percentage / 100.0)
            
        elif method == "kelly":
            if risk_reward <= 0:
                kelly_pct = 0.0
            else:
                kelly_pct = win_rate - ((1.0 - win_rate) / risk_reward)
            safe_kelly = max(0.0, min(kelly_pct / 2.0, self.strategy_settings.risk_percentage / 100.0))
            target_capital = capital * safe_kelly * self.strategy_settings.leverage
            target_size = target_capital / entry_price
            
        elif method == "max_dollar_risk":
            max_fiat = self.risk_settings.max_daily_loss_fiat * 0.5  # E.g. risk half the daily loss max
            target_size = max_fiat / risk_per_unit
            
        elif method == "max_portfolio_risk":
            # Just another name for fixed percentage risk of total equity
            target_size = risk_amount / risk_per_unit
            
        else:
            # Default: ATR Risk / Percentage Risk
            target_size = risk_amount / risk_per_unit

        # Apply constraints
        max_units = (self.strategy_settings.max_position_size * self.strategy_settings.leverage) / entry_price
        final_size = min(target_size, max_units)
        
        # Check available cash limit (assuming no margin for simplicity in sizing, leverage handled separately)
        max_affordable = (portfolio.account.cash * self.strategy_settings.leverage) / entry_price
        final_size = min(final_size, max_affordable)

        if final_size < self.strategy_settings.min_lot_size:
            return 0.0

        return round(final_size, 4)
