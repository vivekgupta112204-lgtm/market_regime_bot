"""Portfolio Risk Aggregator.

Aggregates metrics for the entire portfolio state for reporting.
"""

from __future__ import annotations

from typing import Any
from portfolio.portfolio_state import PortfolioState
from config.settings import RiskSettings


class PortfolioRisk:
    """Aggregates portfolio-level risk metrics."""

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    def get_metrics(self, portfolio: PortfolioState) -> dict[str, Any]:
        """Calculate and return all high-level risk metrics.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Dictionary of metrics.
        """
        equity = portfolio.total_equity
        drawdown = portfolio.account.get_drawdown(equity)
        exposure = portfolio.total_exposure
        open_risk = portfolio.total_open_risk

        return {
            "total_equity": equity,
            "cash": portfolio.account.cash,
            "used_margin": portfolio.account.used_margin,
            "realized_pnl": portfolio.account.realized_pnl,
            "unrealized_pnl": equity - portfolio.account.total_equity,
            "current_drawdown_pct": drawdown,
            "total_exposure_fiat": exposure,
            "exposure_pct": (exposure / equity) if equity > 0 else 0.0,
            "total_open_risk_fiat": open_risk,
            "open_risk_pct": (open_risk / equity) if equity > 0 else 0.0,
            "open_positions_count": len(portfolio.positions),
        }
