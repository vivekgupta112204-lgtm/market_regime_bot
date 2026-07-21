"""Subordinate Agent for Risk Management."""

import numpy as np
from typing import Dict, Any

class RiskAgent:
    """Evaluates institutional risk metrics: VaR, Portfolio Exposure, Drawdowns."""
    
    def __init__(self, max_drawdown_limit: float = 0.15, max_var_limit: float = 0.05, max_leverage: float = 2.0):
        self.max_dd = max_drawdown_limit
        self.max_var = max_var_limit
        self.max_leverage = max_leverage

    def compute_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Historical Value at Risk."""
        if len(returns) == 0: return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    def compute_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        var = self.compute_var(returns, confidence)
        sub_var = returns[returns <= var]
        return float(np.mean(sub_var)) if len(sub_var) > 0 else var

    def check_exposure(self, requested_allocations: Dict[str, float], historical_returns: np.ndarray, current_drawdown: float) -> dict:
        """Returns complex dictionary defining if the requested portfolio lies within risk limits."""
        total_exposure = sum(abs(v) for v in requested_allocations.values())
        
        var_95 = self.compute_var(historical_returns)
        cvar_95 = self.compute_cvar(historical_returns)
        
        flags = []
        if total_exposure > self.max_leverage: flags.append(f"Leverage {total_exposure} exceeds {self.max_leverage}")
        if abs(var_95) > self.max_var: flags.append(f"VaR {abs(var_95)} exceeds {self.max_var}")
        if current_drawdown > self.max_dd: flags.append(f"Drawdown {current_drawdown} exceeds {self.max_dd}")
        
        return {
            "approved": len(flags) == 0,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "total_leverage": total_exposure,
            "violations": flags
        }
