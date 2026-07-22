"""Subordinate Agent for Risk Management (Intraday)."""

import numpy as np
from typing import Dict, Any

class RiskAgent:
    """Evaluates strict intraday risk metrics: Daily Loss, Max Trades, Position Sizing."""
    
    def __init__(self, max_daily_loss: float = 0.02, max_trades_per_day: int = 5, risk_per_trade: float = 0.01):
        self.max_daily_loss = max_daily_loss
        self.max_trades = max_trades_per_day
        self.risk_per_trade = risk_per_trade

    def check_exposure(self, requested_allocations: Dict[str, float], todays_pnl: float, num_trades_today: int, account_size: float) -> dict:
        """Enforces intraday trading constraints."""
        
        flags = []
        
        # 1. Check Max Daily Loss constraint
        if todays_pnl < 0:
            loss_pct = abs(todays_pnl) / account_size
            if loss_pct >= self.max_daily_loss:
                flags.append(f"Max Daily Loss {loss_pct*100:.1f}% exceeded limit {self.max_daily_loss*100}%")
                
        # 2. Check Overtrading constraint
        if num_trades_today >= self.max_trades:
            flags.append(f"Max trades ({self.max_trades}) for the day reached.")
            
        # 3. Size constraints per trade
        max_size = max(abs(v) for v in requested_allocations.values()) if requested_allocations else 0
        if max_size > self.risk_per_trade * 10:
            flags.append("Allocation size suspiciously high for intraday risk limit.")
        
        return {
            "approved": len(flags) == 0,
            "todays_pnl_pct": (todays_pnl / account_size) * 100 if account_size > 0 else 0,
            "trades_count": num_trades_today,
            "violations": flags
        }
