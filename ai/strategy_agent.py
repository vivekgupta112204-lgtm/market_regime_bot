"""Subordinate Agent for Dynamic Intraday Strategy Generation."""

import pandas as pd
import numpy as np

class StrategyAgent:
    """Responsible for automatic feature discovery and intraday rule generation."""
    
    def formulate_new_strategy(self, price_data: pd.DataFrame) -> dict:
        """Evaluates ORB, VWAP, and EMA Crossover for optimal intraday signals."""
        if price_data.empty or "Close" not in price_data.columns or "Volume" not in price_data.columns: 
            return {"new_strategy_discovered": False}
            
        best_spread = 0.0
        best_strat = "Trend Following"
        
        close = price_data['Close']
        high = price_data['High']
        low = price_data['Low']
        vol = price_data['Volume']
        
        # 1. EMA Crossover
        ema9 = close.ewm(span=9).mean().iloc[-1]
        ema21 = close.ewm(span=21).mean().iloc[-1]
        crossover_val = (ema9 - ema21) / ema21
        
        # 2. VWAP Logic
        typical_price = (high + low + close) / 3
        vwap = (typical_price * vol).cumsum() / vol.cumsum()
        current_vwap = vwap.iloc[-1]
        vwap_spread = (close.iloc[-1] - current_vwap) / current_vwap
        
        # 3. Opening Range Breakout (ORB) Approximation (using rolling 3 periods as proxy for first 15m)
        orb_high = high.rolling(3).max().iloc[-2]
        orb_low = low.rolling(3).min().iloc[-2]
        
        if close.iloc[-1] > orb_high:
            orb_signal = 0.01
        elif close.iloc[-1] < orb_low:
            orb_signal = -0.01
        else:
            orb_signal = 0.0
            
        # Strategy selection
        strategies = {
            "EMA Crossover": crossover_val,
            "VWAP Breakout": vwap_spread,
            "ORB Strategy": orb_signal
        }
        
        for name, edge in strategies.items():
            if abs(edge) > abs(best_spread):
                best_spread = edge
                direction = "Bullish" if edge > 0 else "Bearish"
                best_strat = f"{name} {direction}"
                
        return {
            "new_strategy_discovered": True,
            "best_strategy": best_strat,
            "predicted_edge": float(best_spread)
        }
