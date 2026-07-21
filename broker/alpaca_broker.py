"""Alpaca Broker Implementation Stub.

Requires the `alpaca-trade-api` or `alpaca-py` library to be fully functional.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from broker.broker_interface import BrokerInterface
from config.settings import APIKeys
from execution.order_manager import Order, OrderStatus


class AlpacaBroker(BrokerInterface):
    """Alpaca REST API integration for US Equities & Crypto."""

    def __init__(self, api_keys: APIKeys) -> None:
        self.api_keys = api_keys
        self._connected = False

    def connect(self) -> bool:
        if not self.api_keys.alpaca_api_key or not self.api_keys.alpaca_secret_key:
            logger.error("AlpacaBroker: Missing API credentials.")
            return False
            
        logger.info("AlpacaBroker: Connecting to Alpaca endpoints (Stub)...")
        self._connected = True
        return True

    def submit_order(self, order: Order) -> Order:
        if not self._connected:
            order.update_status(OrderStatus.REJECTED)
            return order
            
        logger.info(f"AlpacaBroker: Submitting {order.side.value} order for {order.symbol} (Stub)")
        # Real implementation would map `order.order_type` to Alpaca's OrderClass,
        # post to /v2/orders, and parse the returned JSON to extract the Alpaca order ID.
        order.broker_id = "alpaca_stub_id"
        order.update_status(OrderStatus.SUBMITTED)
        return order

    def cancel_order(self, order_id: str) -> bool:
        logger.info(f"AlpacaBroker: Cancelling order {order_id} (Stub)")
        return True

    def fetch_order_status(self, order: Order) -> Order:
        logger.debug(f"AlpacaBroker: Fetching status for {order.broker_id} (Stub)")
        # Real implementation queries /v2/orders/{id} and updates fill status.
        return order

    def get_account_balance(self) -> dict[str, float]:
        # Real implementation queries /v2/account
        return {"cash": 0.0, "equity": 0.0, "margin_used": 0.0}

    def get_open_positions(self) -> list[dict[str, Any]]:
        # Real implementation queries /v2/positions
        return []
