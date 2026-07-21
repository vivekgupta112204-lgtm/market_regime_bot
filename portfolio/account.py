"""Account equity and available cash tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Account:
    """Represents the broker account balances.

    Attributes:
        initial_capital: The starting capital of the account.
        cash: Currently available un-invested cash.
        used_margin: Capital currently tied up in open positions.
        realized_pnl: Cumulative realized profit and loss.
        high_water_mark: Highest recorded total equity.
    """

    initial_capital: float
    cash: float
    used_margin: float = 0.0
    realized_pnl: float = 0.0
    high_water_mark: float = 0.0

    def __post_init__(self) -> None:
        if self.high_water_mark == 0.0:
            self.high_water_mark = self.initial_capital

    @property
    def total_equity(self) -> float:
        """Total equity is available cash + used margin (plus unrealized, typically handled higher up)."""
        return self.cash + self.used_margin

    def update_high_water_mark(self, current_total_equity: float) -> None:
        """Update the high water mark if current equity exceeds it."""
        if current_total_equity > self.high_water_mark:
            self.high_water_mark = current_total_equity

    def get_drawdown(self, current_total_equity: float) -> float:
        """Calculate the current drawdown percentage from the high water mark.
        
        Returns:
            A positive float representing drawdown percentage (e.g., 0.05 for 5% DD).
        """
        if self.high_water_mark <= 0:
            return 0.0
        dd = (self.high_water_mark - current_total_equity) / self.high_water_mark
        return max(0.0, dd)

    def allocate_margin(self, amount: float) -> bool:
        """Attempt to allocate cash to margin.

        Returns:
            True if sufficient cash is available, False otherwise.
        """
        if amount > self.cash:
            return False
        self.cash -= amount
        self.used_margin += amount
        return True

    def release_margin(self, amount: float, pnl: float) -> None:
        """Release margin back to cash and record realized PnL."""
        self.used_margin -= amount
        # Cash receives the original margin +/- the PnL
        self.cash += (amount + pnl)
        self.realized_pnl += pnl

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "used_margin": self.used_margin,
            "realized_pnl": self.realized_pnl,
            "total_equity": self.total_equity,
            "high_water_mark": self.high_water_mark,
        }
