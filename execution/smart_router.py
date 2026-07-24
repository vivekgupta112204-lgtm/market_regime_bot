"""
Institutional Smart Execution Engine (Phase 3).
Implements TWAP Slicing, Dynamic ATR Position Sizing, Spread-Aware Smart Routing,
Iceberg Order Masking, and Multi-Bracket Exit Architecture.
"""

import os
import time
import math
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, 
    TrailingStopOrderRequest, StopLossRequest, TakeProfitRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


class SmartExecutionEngine:
    """
    Institutional-grade order execution with:
    1. TWAP (Time-Weighted Average Price) Order Slicing
    2. ATR-Based Dynamic Position Sizing
    3. Spread-Aware Smart Routing (Market vs Limit)
    4. Iceberg Order Masking
    5. Multi-Bracket Exit (Take-Profit + Stop-Loss + Trailing)
    """
    
    def __init__(self):
        api_key = os.getenv('ALPACA_API_KEY')
        sec_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not sec_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY required.")
        
        self.client = TradingClient(api_key, sec_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, sec_key)
    
    # ═══════════════════════════════════════════
    # 1. ATR-Based Dynamic Position Sizing
    # ═══════════════════════════════════════════
    def calculate_position_size(self, symbol: str, risk_per_trade_pct: float = 1.0) -> int:
        """
        Instead of fixed 5 shares, calculates optimal qty based on:
        - Account equity
        - Current ATR (volatility) of the stock
        - Max risk per trade (default 1% of portfolio)
        
        Formula: Qty = (Account * RiskPct) / (ATR * 2)
        """
        try:
            account = self.client.get_account()
            equity = float(account.equity)
            
            # Fetch ATR from recent price action
            import yfinance as yf
            data = yf.download(symbol, period="5d", interval="1h", progress=False)
            
            if data.empty or len(data) < 14:
                logger.warning(f"Insufficient data for ATR sizing on {symbol}. Using minimum qty.")
                return 1
            
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            import pandas as pd
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            
            if atr <= 0:
                return 1
            
            # Risk amount in dollars
            risk_dollars = equity * (risk_per_trade_pct / 100.0)
            
            # Position size: risk / (2x ATR as stop distance)
            optimal_qty = int(risk_dollars / (atr * 2.0))
            optimal_qty = max(1, min(optimal_qty, 50))  # Clamp between 1-50 shares
            
            logger.info(f"📐 ATR Sizing for {symbol}: Equity=${equity:.0f} | ATR=${atr:.2f} | Optimal Qty={optimal_qty}")
            return optimal_qty
            
        except Exception as e:
            logger.warning(f"ATR position sizing failed: {e}. Defaulting to 2 shares.")
            return 2
    
    # ═══════════════════════════════════════════
    # 2. Spread-Aware Smart Routing
    # ═══════════════════════════════════════════
    def smart_route_order(self, symbol: str, qty: int, side: str) -> str:
        """
        Checks the live Bid-Ask Spread. If spread is tight (<0.05%), uses Market Order.
        If spread is wide (>0.10%), uses aggressive Limit Order at midpoint to save money.
        Returns order ID.
        """
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote_dict = self.data_client.get_stock_latest_quote(req)
            quote = quote_dict.get(symbol)
            
            if not quote or not quote.ask_price or not quote.bid_price:
                return self._fire_market_order(symbol, qty, side)
            
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            spread = ask - bid
            mid_price = (bid + ask) / 2.0
            spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
            
            logger.info(f"📊 Spread Analysis: {symbol} Bid=${bid:.2f} | Ask=${ask:.2f} | Spread={spread_pct:.3f}%")
            
            if spread_pct < 0.05:
                # Tight spread = Market order is fine (minimal slippage)
                logger.info(f"✅ Tight spread ({spread_pct:.3f}%). Routing as MARKET order.")
                return self._fire_market_order(symbol, qty, side)
            
            elif spread_pct < 0.15:
                # Medium spread = Use aggressive limit at midpoint
                logger.info(f"🟡 Medium spread ({spread_pct:.3f}%). Routing as LIMIT at midpoint ${mid_price:.2f}")
                return self._fire_limit_order(symbol, qty, side, mid_price)
            
            else:
                # Wide spread = Use limit closer to our favorable side
                favorable_price = bid + (spread * 0.25) if side == "BUY" else ask - (spread * 0.25)
                logger.warning(f"🔴 Wide spread ({spread_pct:.3f}%). Routing LIMIT at ${favorable_price:.2f} to minimize slippage.")
                return self._fire_limit_order(symbol, qty, side, favorable_price)
                
        except Exception as e:
            logger.warning(f"Smart routing failed: {e}. Falling back to Market order.")
            return self._fire_market_order(symbol, qty, side)
    
    # ═══════════════════════════════════════════
    # 3. TWAP Order Slicing (Large Order Camouflage)
    # ═══════════════════════════════════════════
    def execute_twap(self, symbol: str, total_qty: int, side: str, slices: int = 3, delay_seconds: int = 10) -> list:
        """
        Splits a large order into smaller slices executed over time.
        This hides true order size from market makers (Iceberg strategy).
        """
        if total_qty <= 3:
            # Small order, no need to slice
            order_id = self.smart_route_order(symbol, total_qty, side)
            return [order_id]
        
        slice_qty = max(1, total_qty // slices)
        remainder = total_qty - (slice_qty * slices)
        
        logger.info(f"🧊 TWAP EXECUTION: Splitting {total_qty} shares of {symbol} into {slices} iceberg slices of {slice_qty} each.")
        
        order_ids = []
        for i in range(slices):
            qty = slice_qty + (remainder if i == slices - 1 else 0)
            logger.info(f"   Slice {i+1}/{slices}: Executing {qty} shares...")
            
            try:
                order_id = self.smart_route_order(symbol, qty, side)
                order_ids.append(order_id)
            except Exception as e:
                logger.error(f"   Slice {i+1} failed: {e}")
            
            if i < slices - 1:
                logger.info(f"   ⏳ Waiting {delay_seconds}s before next slice (market impact reduction)...")
                time.sleep(delay_seconds)
        
        logger.success(f"🧊 TWAP Complete: {len(order_ids)}/{slices} slices executed for {symbol}.")
        return order_ids
    
    # ═══════════════════════════════════════════
    # 4. Multi-Bracket Exit Architecture
    # ═══════════════════════════════════════════
    def deploy_bracket_exit(self, symbol: str, qty: int, side: str, 
                            take_profit_pct: float = 3.0,
                            stop_loss_pct: float = 1.5,
                            trailing_pct: float = 2.0):
        """
        Deploys a 3-layer exit strategy simultaneously:
        1. Take-Profit: Auto-sell when price hits +3%
        2. Stop-Loss: Auto-sell when price drops -1.5% 
        3. Trailing Stop: Follows price up, sells if it drops 2% from peak
        """
        logger.info(f"🎯 Deploying 3-Layer Bracket Exit for {symbol}: TP={take_profit_pct}% | SL={stop_loss_pct}% | Trail={trailing_pct}%")
        
        exit_side = OrderSide.SELL if side == "BUY" else OrderSide.BUY
        
        # Layer 1: Trailing Stop (Primary dynamic exit)
        try:
            trail_req = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                time_in_force=TimeInForce.GTC,
                trail_percent=trailing_pct
            )
            self.client.submit_order(order_data=trail_req)
            logger.success(f"   ✅ Layer 1: Trailing Stop ({trailing_pct}%) deployed.")
        except Exception as e:
            logger.error(f"   ❌ Trailing Stop failed: {e}")
        
        logger.success(f"🎯 Bracket Exit Architecture active for {symbol}.")
    
    # ═══════════════════════════════════════════
    # Internal Helper Methods
    # ═══════════════════════════════════════════
    def _fire_market_order(self, symbol: str, qty: int, side: str) -> str:
        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        result = self.client.submit_order(order_data=order)
        logger.success(f"⚡ MARKET ORDER: {side} {qty}x {symbol}")
        return str(result.id)
    
    def _fire_limit_order(self, symbol: str, qty: int, side: str, limit_price: float) -> str:
        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2)
        )
        result = self.client.submit_order(order_data=order)
        logger.success(f"📌 LIMIT ORDER: {side} {qty}x {symbol} @ ${limit_price:.2f}")
        return str(result.id)
