"""Portfolio Manager.

Orchestrates the various sub-managers and synchronizes local state with broker state.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from portfolio.portfolio_state import PortfolioState
from portfolio.cash_manager import CashManager
from portfolio.position_manager import PositionManager
from portfolio.allocation_manager import AllocationManager
from portfolio.performance_tracker import PerformanceTracker
from broker.broker_interface import BrokerInterface


class PortfolioManager:
    """Central orchestrator for portfolio tracking and accounting."""

    def __init__(self, state: PortfolioState, broker: BrokerInterface | None = None) -> None:
        self.state = state
        self.broker = broker
        
        # Instantiate sub-managers wrapping the state
        self.cash_manager = CashManager(state.account)
        self.position_manager = PositionManager(state)
        self.allocation_manager = AllocationManager(state)
        self.performance_tracker = PerformanceTracker()

    def sync_with_broker(self) -> None:
        """Fetch latest state from broker and synchronize local state.
        
        Provides basic recovery and alignment.
        """
        if not self.broker:
            logger.warning("PortfolioManager: No broker attached for sync.")
            return
            
        balances = self.broker.get_account_balance()
        if "cash" in balances:
            self.state.account.cash = balances["cash"]
            
        # In a full implementation, we'd iterate over broker open positions 
        # and reconcile them against self.state.positions.

    def update_prices(self, current_prices: dict[str, float]) -> None:
        """Update mark-to-market prices for all positions."""
        self.position_manager.update_prices(current_prices)

    def record_trade_closure(self, symbol: str, exit_price: float, quantity: float) -> None:
        """Process a closed position, applying PnL to cash and tracking performance."""
        pos = self.position_manager.remove_position(symbol)
        if not pos:
            return
            
        # Calculate final PnL
        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * quantity
        else:
            pnl = (pos.entry_price - exit_price) * quantity
            
        # Release margin and apply PnL
        original_margin = pos.entry_price * quantity
        self.cash_manager.release_funds(original_margin, pnl)
        
        # Record for performance stats
        self.performance_tracker.record_trade(pnl)
        logger.info(f"PortfolioManager: Closed {symbol} for PnL: {pnl:.2f}")

    def get_summary(self) -> dict[str, Any]:
        """Return high-level summary of the portfolio."""
        return {
            "total_equity": self.state.total_equity,
            "cash": self.state.account.cash,
            "open_positions": len(self.state.positions),
            "performance": self.performance_tracker.get_metrics(),
        }
