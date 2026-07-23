"""Statistical Arbitrage (Pairs Trading) Execution Engine."""

import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger

class StatArbAgent:
    """Calculates spread divergence on historically co-integrated pairs using Z-Scores."""
    
    def __init__(self):
        # Pairs proven to hold strong mathematical co-integration (Mean reverting)
        self.pairs = [
            ("V", "MA"),       # Payments
            ("KO", "PEP"),     # Consumer staples
            ("CVX", "XOM"),    # Energy
            ("META", "GOOGL")  # Tech Ads
        ]
        
    def calculate_z_score(self, asset_a: str, asset_b: str) -> tuple[float, float, float]:
        """Fetches 30 days of 1-hour interval data to compute current rolling Z-Score of the ratio."""
        try:
            # Download synchronized closing data
            data = yf.download(f"{asset_a} {asset_b}", period="1mo", interval="1h", progress=False)['Close']
            data = data.dropna()
            
            if data.empty or len(data) < 20: 
                return 0.0, 0.0, 0.0
                
            # Ratio of Asset A / Asset B
            ratio = data[asset_a] / data[asset_b]
            
            # Simple Z-Score logic using 20-period rolling basis
            rolling_mean = ratio.rolling(window=20).mean()
            rolling_std = ratio.rolling(window=20).std()
            z_scores = (ratio - rolling_mean) / rolling_std
            
            current_z = z_scores.iloc[-1]
            return float(current_z), float(data[asset_a].iloc[-1]), float(data[asset_b].iloc[-1])
        except Exception as e:
            logger.error(f"Stat-Arb failure calculating {asset_a}/{asset_b}: {e}")
            return 0.0, 0.0, 0.0

    def generate_arbitrage_signals(self) -> list[dict]:
        """Identifies pairs with heavy divergence (Z-Score > 2.0 or < -2.0)."""
        signals = []
        logger.info("Stat-Arb Agent initiating pair divergence sweep...")
        
        for asset_a, asset_b in self.pairs:
            z_score, price_a, price_b = self.calculate_z_score(asset_a, asset_b)
            
            if z_score > 2.0:
                logger.success(f"Stat-Arb Anomaly Detected! {asset_a}/{asset_b} Z-Score = {z_score:.2f} (Overvalued {asset_a})")
                signals.append({
                    "pair_id": f"{asset_a}_{asset_b}",
                    "short_target": asset_a,
                    "long_target": asset_b,
                    "z_score": z_score
                })
            elif z_score < -2.0:
                logger.success(f"Stat-Arb Anomaly Detected! {asset_a}/{asset_b} Z-Score = {z_score:.2f} (Undervalued {asset_a})")
                signals.append({
                    "pair_id": f"{asset_a}_{asset_b}",
                    "short_target": asset_b,  # asset B is overvalued relative to A
                    "long_target": asset_a,
                    "z_score": z_score
                })
            else:
                logger.debug(f"Pair {asset_a}/{asset_b} perfectly correlated (Z: {z_score:.2f}). No arb opportunity.")
                
        return signals
