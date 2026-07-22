"""Subordinate Agent for Risk Management (Swing/Positional Trading)."""

import numpy as np
from typing import Dict, Any

class RiskAgent:
    """Evaluates trailing stops and limits for long-term holding."""
    
    def __init__(self, trailing_stop_pct: float = 0.05, max_risk_per_trade: float = 0.02):
        self.trailing_stop = trailing_stop_pct
        self.max_trade_risk = max_risk_per_trade

    def check_exposure(self, requested_allocations: Dict[str, float], todays_pnl: float, num_trades_today: int, account_size: float) -> dict:
        """Enforces Swing Positional Constraints (Does not force close at EOD)."""
        
        flags = []
            
        # Size constraints per trade
        max_size = max(abs(v) for v in requested_allocations.values()) if requested_allocations else 0
        if max_size > self.max_trade_risk * 10:
            flags.append("Allocation size suspiciously high for swing risk limit.")
        
        return {
            "approved": len(flags) == 0,
            "todays_pnl_pct": (todays_pnl / account_size) * 100 if account_size > 0 else 0,
            "violations": flags
        }
