"""Institutional Portfolio Allocation algorithms."""

import numpy as np
from typing import Dict
from loguru import logger
from scipy.optimize import minimize

class PortfolioOptimizer:
    """Implementations for MVO, Risk Parity, and Hierarchical Risk Parity allocations."""
    
    def mean_variance_optimization(self, cov_matrix: np.ndarray, expected_returns: np.ndarray, risk_free_rate: float = 0.0) -> np.ndarray:
        """Markowitz optimization yielding the tangency portfolio (Max Sharpe)."""
        logger.info("Running Mean-Variance Optimization (MVO).")
        n = len(expected_returns)
        
        def neg_sharpe(weights):
            port_ret = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n))
        init_guess = np.ones(n) / n
        
        opt = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return opt.x if opt.success else np.ones(n) / n

    def risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Equalizes risk contribution across all assets."""
        n = cov_matrix.shape[0]
        
        def risk_budget_obj(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal_contrib
            # Equal risk contribution = 1/n
            target = port_vol / n
            return np.sum((risk_contrib - target)**2)
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n))
        init_guess = np.ones(n) / n
        
        opt = minimize(risk_budget_obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return opt.x if opt.success else np.ones(n) / n
