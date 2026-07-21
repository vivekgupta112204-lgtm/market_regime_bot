"""Portfolio state tracking for open positions and active risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from portfolio.account import Account


@dataclass
class OpenPosition:
    """Represents an active trade in the market."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    current_price: float = 0.0

    @property
    def value(self) -> float:
        """Current fiat value of the position."""
        return self.size * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized PnL based on direction."""
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def risk_amount(self) -> float:
        """Fial value at risk if stop loss is hit."""
        return abs(self.entry_price - self.stop_loss) * self.size


@dataclass
class PortfolioState:
    """Aggregates the account and all open positions for risk evaluation.

    Attributes:
        account: The underlying ``Account`` object tracking cash/margin.
        positions: Dictionary mapping symbol to ``OpenPosition``.
    """
    account: Account
    positions: dict[str, OpenPosition] = field(default_factory=dict)

    @property
    def total_equity(self) -> float:
        """Total equity including cash and mark-to-market positions."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.account.total_equity + unrealized

    @property
    def total_exposure(self) -> float:
        """Total fiat value of all open positions."""
        return sum(p.value for p in self.positions.values())

    @property
    def total_open_risk(self) -> float:
        """Total fiat risk if all stop losses are hit simultaneously."""
        return sum(p.risk_amount for p in self.positions.values())

    def update_prices(self, current_prices: dict[str, float]) -> None:
        """Update the mark-to-market prices of open positions."""
        for sym, price in current_prices.items():
            if sym in self.positions:
                self.positions[sym].current_price = price
        
        # After prices update, update HWM on account
        self.account.update_high_water_mark(self.total_equity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account": self.account.to_dict(),
            "open_positions": len(self.positions),
            "total_exposure": self.total_exposure,
            "total_open_risk": self.total_open_risk,
            "total_equity": self.total_equity,
        }
