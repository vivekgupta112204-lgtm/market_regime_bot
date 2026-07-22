"""American US Intraday Stock Scanner Module."""

import time
import pandas as pd
import yfinance as yf
from loguru import logger
import numpy as np

class USIntradayScanner:
    """Scans and filters US Equities every morning."""
    
    def __init__(self, symbols_list: list[str] = None):
        if not symbols_list:
            # Liquid Mega-Cap US Tech Stocks
            self.symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "QQQ"]
        else:
            self.symbols = symbols_list

    def scan_morning_opportunities(self) -> list[str]:
        """Runs the morning scan for high momentum Swing Trading opportunities."""
        logger.info("Initiating Pre-Market US Scan for Positional/Swing Opportunities...")
        top_picks = []
        
        try:
            # Downloading 1 Hour interval data for the last 1 Month
            df = yf.download(self.symbols, period="1mo", interval="1h", group_by="ticker", progress=False)
            
            for symbol in self.symbols:
                try:
                    if len(self.symbols) == 1:
                        data = df
                    else:
                        data = df[symbol]
                        
                    if data.empty: continue
                    
                    close = data['Close']
                    vol = data['Volume']
                    
                    if len(close) < 50: continue
                    
                    # Compute EMAs
                    ema20 = close.ewm(span=20).mean().iloc[-1]
                    ema50 = close.ewm(span=50).mean().iloc[-1]
                    
                    # VWAP Approximation
                    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                    vwap = (typical_price * vol).cumsum() / vol.cumsum()
                    current_vwap = vwap.iloc[-1]
                    
                    current_price = close.iloc[-1]
                    current_vol = vol.iloc[-1]
                    avg_vol = vol.rolling(20).mean().iloc[-1]
                    
                    # Constraints
                    if current_price > ema20 and current_price > ema50 and current_price > current_vwap:
                        if current_vol > (avg_vol * 1.2):  # Relative volume > 1.2
                            top_picks.append(symbol)
                except Exception as e:
                    pass
        except Exception as e:
            logger.error(f"Scan strictly failed: {e}")
            
        logger.info(f"US Market Scan Complete. Found {len(top_picks)} intraday opportunities.")
        return top_picks[:20]
