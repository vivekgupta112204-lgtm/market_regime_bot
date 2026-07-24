import sys
import os
import asyncio
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live.crypto import CryptoDataStream
from numba import njit

# 1. Setup Logging
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>CRYPTO-HFT-ENGINE</cyan> - <white>{message}</white>"
logger.add(sys.stdout, format=log_format, level="INFO")
logger.add("logs/crypto_hft_logs.txt", rotation="50 MB")

API_KEY = os.getenv('ALPACA_API_KEY')
SEC_KEY = os.getenv('ALPACA_SECRET_KEY')

# Fast C-Compiled Math (Bypass GIL)
@njit(nogil=True)
def calculate_nano_momentum_c(prices):
    if len(prices) < 2:
        return 0.0
    return prices[-1] - prices[0]

class CryptoLowLatencyHFT:
    """Micro-second High-Frequency Scalping Engine for Crypto (WebSocket)"""
    def __init__(self, target_symbols):
        self.symbols = target_symbols
        self.tick_windows = {sym: [] for sym in target_symbols}
        
        self.trade_lock = False
        self.trading_allowed = True
        self.daily_target_usd = 50.0  # Common risk cap
        self.daily_loss_limit_usd = -25.0
        
        if not API_KEY or not SEC_KEY:
            logger.critical("Alpaca API keys missing! HFT Crypto requires valid connection.")
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY required.")
            
        self.client = TradingClient(API_KEY, SEC_KEY, paper=True)
        self.stream = CryptoDataStream(API_KEY, SEC_KEY)

    async def _trade_handler(self, data):
        """Callback fired dynamically generated on every crypto trade tick in milliseconds"""
        if self.trade_lock or not self.trading_allowed:
            return
            
        sym = data.symbol
        price = data.price
        
        # Micro tick window per symbol
        self.tick_windows[sym].append(price)
        if len(self.tick_windows[sym]) > 10:
             self.tick_windows[sym].pop(0)
             
             tick_array = np.array(self.tick_windows[sym], dtype=np.float32)
             price_delta = calculate_nano_momentum_c(tick_array)
             
             # Calculate momentum percentage instead of absolute dollars due to different coin values
             momentum_pct = price_delta / price
             
             # HFT Strategy: Nano-Momentum Burst threshold (0.1% sudden spike)
             if momentum_pct > 0.001: 
                 logger.success(f"⚡ [CRYPTO HFT]: Massive Burst (+{momentum_pct*100:.3f}%) in {sym}. FIRE LONG SCALP!")
                 await self._execute_scalp("BUY", sym)
                 self.tick_windows[sym].clear()
                  
             elif momentum_pct < -0.001:
                 logger.error(f"⚡ [CRYPTO HFT]: Plunge (-{abs(momentum_pct*100):.3f}%) in {sym}. FIRE SHORT SCALP!")
                 await self._execute_scalp("SELL", sym)
                 self.tick_windows[sym].clear()

    async def _execute_scalp(self, direction: str, symbol: str):
        self.trade_lock = True
        try:
             order = MarketOrderRequest(
                 symbol=symbol,
                 notional=100.0,  # Strict $100 Fractional sizing
                 side=OrderSide.BUY if direction == "BUY" else OrderSide.SELL,
                 time_in_force=TimeInForce.GTC
             )
             self.client.submit_order(order_data=order)
             logger.success(f"HFT Crypto SUCCESS: {direction} $100 of {symbol}")
             
             # Target Evaluation
             account = self.client.get_account()
             daily_pnl = float(account.equity) - float(account.last_equity)
             logger.info(f"📊 CRYPTO PNL UPDATE: Today's Profit is ${daily_pnl:.2f}")
             
             if daily_pnl >= self.daily_target_usd:
                 logger.success(f"🎯 INSTITUTIONAL CAP REACHED: Profit (+${daily_pnl:.2f}) cap hit. Crypto HFT shutting down till tomorrow.")
                 self.trading_allowed = False
             elif daily_pnl <= self.daily_loss_limit_usd:
                 logger.critical(f"🛑 RISK LIMIT REACHED: Loss (-${abs(daily_pnl):.2f}) cap hit. Crypto HFT halted.")
                 self.trading_allowed = False
                 
        except Exception as e:
             logger.error(f"Crypto HFT Execution failed: {e}")
        finally:
             await asyncio.sleep(2)
             self.trade_lock = False

    def run_engine(self):
        logger.info(f"🚀 Igniting Crypto Millisecond WebSockets for: {self.symbols}")
        for sym in self.symbols:
            self.stream.subscribe_trades(self._trade_handler, sym)
            
        try:
            self.stream.run()
        except KeyboardInterrupt:
            logger.info("Crypto HFT Engine Halted manually.")
        except Exception as e:
            logger.error(f"Crypto HFT Stream Crash: {e}")

if __name__ == "__main__":
    targets = ["BTC/USD", "ETH/USD", "SOL/USD"]
    bot = CryptoLowLatencyHFT(targets)
    bot.run_engine()
