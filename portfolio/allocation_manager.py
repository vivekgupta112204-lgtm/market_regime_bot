"""Allocation Manager.

Manages portfolio weighting rules (e.g. Equal Weight, Fixed Allocation).
"""

from __future__ import annotations

from typing import Any
from portfolio.portfolio_state import PortfolioState


class AllocationManager:
    """Calculates target weights and allocation bounds."""

    def __init__(self, portfolio: PortfolioState, max_weight_pct: float = 0.2) -> None:
        self.portfolio = portfolio
        self.max_weight_pct = max_weight_pct

    def get_max_allowed_fiat(self) -> float:
        """Calculate the absolute maximum fiat size a single position can take."""
        return self.portfolio.total_equity * self.max_weight_pct

    def apply_equal_weight(self, candidates: list[str]) -> dict[str, float]:
        """Generate a target allocation dictionary assuming equal weights for candidates."""
        if not candidates:
            return {}
            
        weight = 1.0 / len(candidates)
        # Cap at max_weight
        weight = min(weight, self.max_weight_pct)
        
        return {sym: weight for sym in candidates}
