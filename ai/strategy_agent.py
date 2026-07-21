"""Subordinate Agent for Dynamic Strategy Generation."""

import pandas as pd
import numpy as np

class StrategyAgent:
    """Responsible for automatic feature discovery and rule generation."""
    
    def formulate_new_strategy(self, price_data: pd.Series, lookbacks: list[int] = [10, 20, 50, 200]) -> dict:
        """Determines best moving average crossover strategy mathematically."""
        if price_data.empty: return {"new_strategy_discovered": False}
        
        best_spread = 0.0
        best_strat = "Trend Following"
        
        # Evaluate multiple moving average spreads to find the strongest momentum factor
        for fast in lookbacks[:2]:
            for slow in lookbacks[2:]:
                fast_ma = price_data.rolling(fast).mean().iloc[-1]
                slow_ma = price_data.rolling(slow).mean().iloc[-1]
                
                spread_pct = (fast_ma - slow_ma) / slow_ma
                
                if abs(spread_pct) > abs(best_spread):
                    best_spread = spread_pct
                    direction = "Bullish Crossover" if spread_pct > 0 else "Bearish Crossover"
                    best_strat = f"Momentum {fast}/{slow} {direction}"
                    
        return {
            "new_strategy_discovered": True,
            "best_strategy": best_strat,
            "predicted_edge": float(best_spread)
        }
