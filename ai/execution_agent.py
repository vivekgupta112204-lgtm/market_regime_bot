"""Subordinate Agent for Trade Execution Optimization."""

class ExecutionAgent:
    """Calculates slippage, limits, and order splitting strategies mathematically."""
    
    def suggest_routes(self, order_size: int, avg_volume: float, atr: float, spread: float, urgency_factor: float = 1.0) -> dict:
        """Dynamically decides TWAP, VWAP, Limit or Market."""
        impact = (order_size / avg_volume) * 100 if avg_volume > 0 else 100
        slippage_est = (spread / 2) + (atr * impact * 0.1)
        
        # Logic constraints mapping 
        if impact < 0.1 and urgency_factor > 1.5:
            rec = "MARKET"
        elif spread > atr * 0.5:
            rec = "LIMIT"
        elif impact > 5.0:
            rec = "VWAP"
        else:
            rec = "TWAP"
            
        prob = max(0.01, 1.0 - (impact / 100.0) - (spread / (atr + 1e-5)))

        return {
            "recommended_order_type": rec,
            "urgency": "High" if urgency_factor > 1.2 else "Low",
            "estimated_slippage_bps": float(slippage_est * 10000),
            "estimated_fill_probability": float(prob),
            "market_impact_pct": impact
        }
