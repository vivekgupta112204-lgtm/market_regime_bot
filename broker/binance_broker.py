"""Binance Broker Implementation Stub.

Requires the `binance-connector` library to be fully functional.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from broker.broker_interface import BrokerInterface
from config.settings import APIKeys
from execution.order_manager import Order, OrderStatus


class BinanceBroker(BrokerInterface):
    """Binance Spot/Futures API integration."""

    def __init__(self, api_keys: APIKeys) -> None:
        self.api_keys = api_keys
        self._connected = False

    def connect(self) -> bool:
        logger.info("BinanceBroker: Connecting (Stub)...")
        self._connected = True
        return True

    def submit_order(self, order: Order) -> Order:
        logger.info(f"BinanceBroker: Submitting order for {order.symbol} (Stub)")
        order.broker_id = "binance_stub_id"
        order.update_status(OrderStatus.SUBMITTED)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def fetch_order_status(self, order: Order) -> Order:
        return order

    def get_account_balance(self) -> dict[str, float]:
        return {"cash": 0.0, "equity": 0.0, "margin_used": 0.0}

    def get_open_positions(self) -> list[dict[str, Any]]:
        return []
