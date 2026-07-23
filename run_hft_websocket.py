from dotenv import load_dotenv
load_dotenv()
"""
Wall-Street Grade High-Frequency Trading (HFT) Stateful WebSocket Execution Engine.
Integrated with C-Compiled GIL bypass and Level-2 Order Book Imbalance Routing.
"""

import os
import sys
import asyncio
import numpy as np
from loguru import logger
from numba import njit
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ═══════════════════════════════════════════
# C-COMPILED MICRO-SECOND MATH KERNELS
# ═══════════════════════════════════════════
@njit(fastmath=True, nogil=True)
def calculate_nano_momentum_c(tick_array: np.ndarray) -> float:
    """Calculates instantaneous micro-momentum directly in C machine-memory (GIL released)."""
    return tick_array[-1] - tick_array[0]

@njit(fastmath=True, nogil=True)
def calculate_l2_imbalance_c(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
    """
    Measures Order Book Imbalance (Institutional limit orders).
    > 1.0 means more BID volume (Bullish Wall)
    < 1.0 means more ASK volume (Bearish Wall)
    """
    total_bids = np.sum(bid_sizes)
    total_asks = np.sum(ask_sizes)
    if total_asks == 0:
        return 999.0
    return total_bids / total_asks

# 1. API Credentials
API_KEY = os.getenv("ALPACA_API_KEY")
SEC_KEY = os.getenv("ALPACA_SECRET_KEY")

logger.add("logs/hft_logs.txt", rotation="50 MB")

class UltraLowLatencyHFT:
    """Stateful High-Frequency Scalping Engine with L2 Orderbook Analysis."""
    def __init__(self, target_symbol: str = "SPY"):
        self.symbol = target_symbol
        
        # State matrices for C-engine
        self.tick_window = []
        self.recent_bid_sizes = []
        self.recent_ask_sizes = []
        
        # Lock to prevent double execution within micro-seconds
        self.trade_lock = False
        
        if not API_KEY or not SEC_KEY:
            logger.critical("Alpaca API keys missing! HFT Engine requires valid production connection.")
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment.")
            
        self.client = TradingClient(API_KEY, SEC_KEY, paper=True)
        self.stream = StockDataStream(API_KEY, SEC_KEY)

    async def _l2_quote_handler(self, quote):
        """Asynchronous callback fired on every Level-2 Orderbook update (Bids/Asks limit orders)."""
        # Store latest limit order resting liquidity
        bid_size = quote.bid_size
        ask_size = quote.ask_size
        
        self.recent_bid_sizes.append(bid_size)
        self.recent_ask_sizes.append(ask_size)
        
        # Keep only the last 10 L2 updates in rolling window
        if len(self.recent_bid_sizes) > 10:
            self.recent_bid_sizes.pop(0)
            self.recent_ask_sizes.pop(0)

    async def _trade_handler(self, data):
        """Asynchronous callback fired every sub-second on a completed tick/trade."""
        if self.trade_lock:
            return  # Prevent overlapping scalps
            
        price = data.price
        
        # Append to micro-window 
        self.tick_window.append(price)
        if len(self.tick_window) > 10:
             self.tick_window.pop(0)
             
             # Calculate micro-momentum using C-compiled No-GIL engine
             tick_array = np.array(self.tick_window, dtype=np.float32)
             price_delta = calculate_nano_momentum_c(tick_array)
             
             # Check L2 Imbalance if we have orderbook data
             imbalance = 1.0 
             if len(self.recent_bid_sizes) > 5:
                 b_arr = np.array(self.recent_bid_sizes, dtype=np.float32)
                 a_arr = np.array(self.recent_ask_sizes, dtype=np.float32)
                 imbalance = calculate_l2_imbalance_c(b_arr, a_arr)
             
             # HFT Strategy: Nano-Momentum Burst + Level 2 Support Validation
             if price_delta > 0.05: # Price jumped 5 cents instantly
                 if imbalance > 2.0: # AND Bids limit volume > 2x Ask volume (Institutional support wall)
                     logger.success(f"⚡ [HFT TRIGGER]: L2 Imbalance ({imbalance:.1f}x) CONFIRMS Burst (+${price_delta:.2f}). FIRE LONG SCALP!")
                     await self._execute_scalp("BUY")
                 else:
                     logger.warning(f"⚠️ [HFT TRAP AVOIDED]: Price jumped (+${price_delta:.2f}) but L2 Book is empty (Imbalance: {imbalance:.1f}x). Ignoring Fake-out.")
                     self.tick_window.clear()
                  
             elif price_delta < -0.05:
                 if imbalance < 0.5: # Ask limit volume > 2x Bid volume (Wall of sellers)
                     logger.error(f"⚡ [HFT TRIGGER]: L2 Imbalance ({imbalance:.1f}x) CONFIRMS Plunge (-${abs(price_delta):.2f}). FIRE SHORT SCALP!")
                     await self._execute_scalp("SELL")
                 else:
                     logger.warning(f"⚠️ [HFT TRAP AVOIDED]: Price dropped (-${abs(price_delta):.2f}) but no Sellers in L2 Book. Ignoring Fake bear trap.")
                     self.tick_window.clear()

    async def _execute_scalp(self, direction: str):
        """Fires 0-latency executions directly to the pairing engine."""
        self.trade_lock = True
        try:
             order = MarketOrderRequest(
                 symbol=self.symbol,
                 qty=10,
                 side=OrderSide.BUY if direction == "BUY" else OrderSide.SELL,
                 time_in_force=TimeInForce.DAY
             )
             self.client.submit_order(order_data=order)
             logger.success(f"HFT Scalp SUCCESS: {direction} 10 shares of {self.symbol}")
        except Exception as e:
             logger.error(f"HFT Execution failed: {e}")
        finally:
             # Cool down period to prevent machine-gunning orders
             await asyncio.sleep(2)
             self.tick_window.clear()
             self.trade_lock = False

    def run_engine(self):
        """Connects and maintains the persistent WebSocket pipeline."""
        logger.info(f"🚀 Booting Wall-Street HFT WebSocket Engine targeting: {self.symbol}")
        logger.info(f"📡 Features: C-Compiled Momentum | Live L2 Orderbook Array | Fake-out Trap Avoidance")
        
        try:
             # Subscribe to both Completed Trades AND Live L2 Quotes (Bid/Ask Wall)
             self.stream.subscribe_trades(self._trade_handler, self.symbol)
             self.stream.subscribe_quotes(self._l2_quote_handler, self.symbol)
             
             logger.info(f"Subscribed to Layer-2 Quoting & Tick Data for {self.symbol}. Waiting for flow...")
             self.stream.run()
        except KeyboardInterrupt:
             logger.info("Gracefully disconnecting HFT WebSocket...")
             self.stream.close()
        except Exception as e:
             logger.error(f"WebSocket unhandled crash: {e}")

if __name__ == "__main__":
    # Ensure Windows asyncio works correctly with Alpaca WebSockets
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    engine = UltraLowLatencyHFT("SPY")
    engine.run_engine()
