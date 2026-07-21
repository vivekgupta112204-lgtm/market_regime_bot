"""Risk Report Generator.

Generates structured reports for humans and logging systems.
"""

from __future__ import annotations

import json
from typing import Any

from risk.portfolio_risk import PortfolioRisk
from portfolio.portfolio_state import PortfolioState


class RiskReport:
    """Generates formatted risk reports."""

    def __init__(self, portfolio_risk: PortfolioRisk) -> None:
        self.portfolio_risk = portfolio_risk

    def generate_report(self, portfolio: PortfolioState) -> dict[str, Any]:
        """Generate a complete dictionary report of the portfolio."""
        metrics = self.portfolio_risk.get_metrics(portfolio)
        
        # Add positions details
        positions_info = []
        for sym, pos in portfolio.positions.items():
            positions_info.append({
                "symbol": sym,
                "direction": pos.direction,
                "size": pos.size,
                "entry": pos.entry_price,
                "current": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "risk_amount": pos.risk_amount,
            })
            
        metrics["positions"] = positions_info
        return metrics

    def generate_json_report(self, portfolio: PortfolioState) -> str:
        """Generate a JSON string report."""
        return json.dumps(self.generate_report(portfolio), indent=4)
