"""Detection engine for extreme unusual Options Activity indicating Institutional Sweeps."""

import yfinance as yf
import pandas as pd
from loguru import logger
from datetime import datetime

class DarkPoolRadar:
    """Scrapes free Option Chains to detect Block/Sweep Whale flow."""
    def __init__(self, unusual_volume_threshold: float = 5.0):
        # Sweeps are defined by Volume massively overwhelming Open Interest by at least 5x
        self.sweep_threshold = unusual_volume_threshold
        
    def scan_unusual_flow(self, symbol: str) -> dict:
        """
        Scans Option Chain and returns a calculated Whale Sentiment.
        > +1.0 = Massive Call Sweeps (Bullish Insiders)
        > -1.0 = Massive Put Sweeps (Bearish Insiders)
        >  0.0 = Normal / Neutral flow
        """
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            if not options_dates:
                return {"whale_sentiment": 0.0, "signal": "NONE", "confidence": 0.0}
            
            # Check the nearest expiration date (most explosive Institutional flows are short-dated DTE)
            near_date = options_dates[0]
            chain = ticker.option_chain(near_date)
            
            calls = chain.calls
            puts = chain.puts
            
            # Filter valid rows to prevent zero division
            calls = calls[(calls['openInterest'] > 0) & (calls['volume'] > 0)].copy()
            puts = puts[(puts['openInterest'] > 0) & (puts['volume'] > 0)].copy()
            
            if calls.empty and puts.empty:
                 return {"whale_sentiment": 0.0, "signal": "NONE", "confidence": 0.0}
                 
            # Calculate Volume / OI Ratio
            calls['sweep_ratio'] = calls['volume'] / calls['openInterest']
            puts['sweep_ratio'] = puts['volume'] / puts['openInterest']
            
            # Find extreme isolated sweeps
            extreme_calls = calls[calls['sweep_ratio'] >= self.sweep_threshold]
            extreme_puts = puts[puts['sweep_ratio'] >= self.sweep_threshold]
            
            total_call_sweep_volume = extreme_calls['volume'].sum()
            total_put_sweep_volume = extreme_puts['volume'].sum()
            
            if total_call_sweep_volume > (total_put_sweep_volume * 1.5) and total_call_sweep_volume > 1000:
                 logger.success(f"WHALE DETECTED: Explosive CALL Sweeps on {symbol} (Vol: {total_call_sweep_volume})")
                 return {"whale_sentiment": 1.0, "signal": "BULL_SWEEP", "confidence": 0.9}
                 
            elif total_put_sweep_volume > (total_call_sweep_volume * 1.5) and total_put_sweep_volume > 1000:
                 logger.error(f"WHALE DETECTED: Explosive PUT Sweeps on {symbol} (Vol: {total_put_sweep_volume})")
                 return {"whale_sentiment": -1.0, "signal": "BEAR_SWEEP", "confidence": 0.9}
            
            # If no massive sweeps detected or they cancel each other out
            return {"whale_sentiment": 0.0, "signal": "NEUTRAL", "confidence": 0.0}
            
        except Exception as e:
            logger.warning(f"DarkPoolRadar failed to scrape Options Chain for {symbol}: {e}")
            return {"whale_sentiment": 0.0, "signal": "NONE", "confidence": 0.0}
