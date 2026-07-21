"""Position Manager.

Advanced tracking of open positions including MAE (Max Adverse Excursion) 
and MFE (Max Favorable Excursion).
Wraps the Phase 5 PortfolioState.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from portfolio.portfolio_state import PortfolioState, OpenPosition


class PositionManager:
    """Manages lifecycle and advanced metrics for open positions."""

    def __init__(self, portfolio: PortfolioState) -> None:
        self.portfolio = portfolio
        # Store advanced metrics separately so we don't need to rewrite the Phase 5 OpenPosition dataclass
        self._mae: dict[str, float] = {}
        self._mfe: dict[str, float] = {}

    def add_position(self, position: OpenPosition) -> None:
        """Add a newly opened position to the portfolio."""
        self.portfolio.positions[position.symbol] = position
        self._mae[position.symbol] = 0.0
        self._mfe[position.symbol] = 0.0
        logger.info(f"PositionManager: Added {position.direction} position in {position.symbol}")

    def remove_position(self, symbol: str) -> OpenPosition | None:
        """Remove a closed position and clean up metrics."""
        pos = self.portfolio.positions.pop(symbol, None)
        self._mae.pop(symbol, None)
        self._mfe.pop(symbol, None)
        if pos:
            logger.info(f"PositionManager: Removed position in {symbol}")
        return pos

    def get_position(self, symbol: str) -> OpenPosition | None:
        return self.portfolio.positions.get(symbol)

    def update_prices(self, current_prices: dict[str, float]) -> None:
        """Update mark-to-market prices and recalculate MAE/MFE."""
        self.portfolio.update_prices(current_prices)
        
        for sym, pos in self.portfolio.positions.items():
            if sym in current_prices:
                price = current_prices[sym]
                
                # Calculate excursion
                if pos.direction == "LONG":
                    favorable = price - pos.entry_price
                    adverse = pos.entry_price - price
                else:
                    favorable = pos.entry_price - price
                    adverse = price - pos.entry_price
                    
                # Update MFE (highest favorable)
                if favorable > self._mfe.get(sym, 0.0):
                    self._mfe[sym] = favorable
                    
                # Update MAE (highest adverse)
                if adverse > self._mae.get(sym, 0.0):
                    self._mae[sym] = adverse

    def get_position_metrics(self, symbol: str) -> dict[str, Any]:
        """Return advanced metrics for a specific position."""
        pos = self.get_position(symbol)
        if not pos:
            return {}
            
        return {
            "symbol": sym,
            "unrealized_pnl": pos.unrealized_pnl,
            "mae": self._mae.get(symbol, 0.0),
            "mfe": self._mfe.get(symbol, 0.0),
        }
