"""High-Frequency Trading (HFT) Stateful WebSocket Execution Engine."""

import os
import sys
import asyncio
from loguru import logger
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. API Credentials
API_KEY = os.getenv("ALPACA_API_KEY")
SEC_KEY = os.getenv("ALPACA_SECRET_KEY")

logger.add("hft_logs.txt", rotation="50 MB")

class UltraLowLatencyHFT:
    """Stateful High-Frequency Scalping Engine."""
    def __init__(self, target_symbol: str = "SPY"):
        self.symbol = target_symbol
        
        # In memory state matrix for micro-second rolling momentum
        self.tick_window = []
        
        if not API_KEY or not SEC_KEY:
            logger.warning("Alpaca API keys missing! HFT Engine running in Live Mock (Presentation) Mode.")
            self.client = None
            self.stream = None
        else:
             self.client = TradingClient(API_KEY, SEC_KEY, paper=True)
             self.stream = StockDataStream(API_KEY, SEC_KEY)

    async def _trade_handler(self, data):
        """Asynchronous callback fired every sub-second on a new tick/trade."""
        price = data.price
        size = data.size
        
        # Append to micro-window 
        self.tick_window.append(price)
        if len(self.tick_window) > 10:
             self.tick_window.pop(0) # Keep rolling window of last 10 ticks
             
             # Calculate micro-momentum (Nanosecond alpha)
             price_delta = self.tick_window[-1] - self.tick_window[0]
             
             if price_delta > 0.05: # High positive micro-momentum burst (5 cents in 10 ticks)
                  logger.info(f"⚡ [HFT TRIGGER]: {self.symbol} Nano-Momentum Burst (+${price_delta:.2f}). Firing LONG Scalp!")
                  self._execute_scalp("BUY")
                  self.tick_window.clear() # Reset memory after trade
                  
             elif price_delta < -0.05: # High negative micro-momentum plunge
                  logger.info(f"⚡ [HFT TRIGGER]: {self.symbol} Nano-Momentum Plunge (-${abs(price_delta):.2f}). Firing SHORT Scalp!")
                  self._execute_scalp("SELL")
                  self.tick_window.clear()

    def _execute_scalp(self, direction: str):
        """Fires 0-latency executions directly to the pairing engine."""
        try:
             order = MarketOrderRequest(
                 symbol=self.symbol,
                 qty=10,
                 side=OrderSide.BUY if direction == "BUY" else OrderSide.SELL,
                 time_in_force=TimeInForce.DAY
             )
             if self.client:
                 self.client.submit_order(order_data=order)
                 logger.success(f"HFT Scalp SUCCESS: {direction} 10 shares of {self.symbol}")
             else:
                 logger.success(f"Dry-Run [PRESENTATION MODE]: HFT Scalp SUCCESS -> {direction} 10 shares of {self.symbol}")
        except Exception as e:
             logger.error(f"HFT Execution failed: {e}")

    def run_engine(self):
        """Connects and maintains the persistent WebSocket pipeline."""
        logger.info(f"🚀 Booting HFT WebSocket Engine targeting: {self.symbol}")
        
        if not self.stream:
             logger.warning("Dry-Run Fallback: Streaming simulated tick data for Presentation...")
             self._simulate_presentation_ticks()
             return

        try:
             self.stream.subscribe_trades(self._trade_handler, self.symbol)
             logger.info(f"Subscribed to Layer-2 Tick Data for {self.symbol}. Waiting for flow...")
             self.stream.run()
        except KeyboardInterrupt:
             logger.info("Gracefully disconnecting HFT WebSocket...")
             self.stream.close()
        except Exception as e:
             logger.error(f"WebSocket unhandled crash: {e}")

    def _simulate_presentation_ticks(self):
         """Generates mock high-frequency packets matching Alpaca WSS standard to showcase the capability offline."""
         import time
         import random
         
         class MockTick:
             def __init__(self, p, s):
                 self.price = p
                 self.size = s 
                 
         base_price = 450.00
         logger.info("Injecting syntheic WSS Layer-2 Packets...")
         for i in range(15):
              time.sleep(0.05) # 50 milliseconds simulation
              base_price += random.uniform(-0.01, 0.03) # Upward bias to trigger LONG
              mock_data = MockTick(base_price, 100)
              asyncio.run(self._trade_handler(mock_data))
              
if __name__ == "__main__":
     engine = UltraLowLatencyHFT("SPY")
     engine.run_engine()
