"""Zerodha Kite Connect Indian Stock Market Broker Adapter."""

import os
from loguru import logger

class ZerodhaBroker:
    """Handles execution on NSE/BSE via Zerodha SmartAPI."""
    
    def __init__(self, api_key: str = None, access_token: str = None):
        self.api_key = api_key or os.getenv("ZERODHA_API_KEY")
        self.access_token = access_token or os.getenv("ZERODHA_ACCESS_TOKEN")
        self.kite = None
        self._initialize()
        
    def _initialize(self):
        try:
            from kiteconnect import KiteConnect
            if self.api_key and self.access_token:
                self.kite = KiteConnect(api_key=self.api_key)
                self.kite.set_access_token(self.access_token)
                logger.info("Zerodha Kite connected successfully.")
        except ImportError:
            logger.warning("kiteconnect package not installed. Zerodha API offline.")

    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "MARKET"):
        """Sends an intraday (MIS) order to NSE."""
        if not self.kite:
            logger.error("Zerodha offline. Cannot execute NSE order.")
            return None
        
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if side.upper() == 'BUY' else self.kite.TRANSACTION_TYPE_SELL,
                quantity=qty,
                product=self.kite.PRODUCT_MIS,  # Intraday Square-off
                order_type=self.kite.ORDER_TYPE_MARKET if order_type == 'MARKET' else self.kite.ORDER_TYPE_LIMIT,
                validity=self.kite.VALIDITY_DAY
            )
            logger.info(f"Zerodha Order Placed: {order_id} ({side} {qty} {symbol})")
            return order_id
        except Exception as e:
            logger.error(f"Zerodha Execution Failed: {e}")
            return None
