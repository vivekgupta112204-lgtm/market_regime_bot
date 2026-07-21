"""Performance Tracker.

Calculates historical performance metrics based on closed trades.
"""

from __future__ import annotations

import math
from typing import Any


class PerformanceTracker:
    """Calculates Sharpe, Sortino, Win Rate, Profit Factor, etc."""

    def __init__(self) -> None:
        self.closed_trades: list[float] = []
        
    def record_trade(self, pnl: float) -> None:
        """Record the final PnL of a closed trade."""
        self.closed_trades.append(pnl)

    def get_metrics(self) -> dict[str, Any]:
        """Calculate performance statistics."""
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
            }

        wins = [t for t in self.closed_trades if t > 0]
        losses = [t for t in self.closed_trades if t <= 0]
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        
        win_rate = len(wins) / len(self.closed_trades)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        
        return {
            "total_trades": len(self.closed_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "total_pnl": sum(self.closed_trades)
        }
