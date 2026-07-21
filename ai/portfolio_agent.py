"""Subordinate Agent for Portfolio Strategy & Balancing."""

import numpy as np
from optimization.portfolio_optimizer import PortfolioOptimizer

class PortfolioAgent:
    """Decides on sizing and diversification."""
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer()
    
    def evaluate_allocation(self, symbols: list[str], expected_returns: np.ndarray, cov_matrix: np.ndarray, max_position_limit: float = 0.5) -> dict:
        """Determines best capital allocation mathematically."""
        if not symbols or len(symbols) == 0:
            return {"suggested_allocations": {}, "risk_ok": False, "portfolio_sharpe": 0.0}
            
        weights = self.optimizer.mean_variance_optimization(cov_matrix, expected_returns)
        
        # Apply limits
        weights = np.clip(weights, 0, max_position_limit)
        weights = weights / np.sum(weights)
        
        allocations = {sym: float(w) for sym, w in zip(symbols, weights)}
        
        port_ret = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_ret / port_vol if port_vol > 0 else 0
        
        return {
            "suggested_allocations": allocations,
            "risk_ok": sharpe > 0.5,
            "portfolio_sharpe": sharpe,
            "portfolio_volatility": float(port_vol)
        }
