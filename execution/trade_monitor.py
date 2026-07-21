"""Trade Monitor.

Tracks pending and active orders, syncing their state from the broker.
"""

from __future__ import annotations

from typing import Callable
from loguru import logger

from broker.broker_interface import BrokerInterface
from execution.order_manager import Order, OrderStatus


class TradeMonitor:
    """Monitors active orders and triggers callbacks on state changes."""

    def __init__(self, broker: BrokerInterface) -> None:
        self.broker = broker
        self._monitored_orders: dict[str, Order] = {}
        # Callbacks triggered when an order is filled, canceled, or rejected
        self._on_fill_callbacks: list[Callable[[Order], None]] = []
        self._on_cancel_callbacks: list[Callable[[Order], None]] = []

    def register_on_fill(self, callback: Callable[[Order], None]) -> None:
        self._on_fill_callbacks.append(callback)

    def register_on_cancel(self, callback: Callable[[Order], None]) -> None:
        self._on_cancel_callbacks.append(callback)

    def monitor_order(self, order: Order) -> None:
        """Add an order to the active monitoring list."""
        if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILL):
            self._monitored_orders[order.id] = order

    def update(self) -> None:
        """Poll the broker for updates on all monitored orders.
        
        In a real event-driven system (like websockets), this would process incoming messages.
        Here we use polling for simplicity and REST API compatibility.
        """
        completed_ids = []
        
        for order_id, order in self._monitored_orders.items():
            old_status = order.status
            updated_order = self.broker.fetch_order_status(order)
            
            if updated_order.status != old_status:
                logger.debug(f"TradeMonitor: Order {order.id} status changed: {old_status.value} -> {updated_order.status.value}")
                
            if updated_order.status == OrderStatus.FILLED:
                for cb in self._on_fill_callbacks:
                    cb(updated_order)
                completed_ids.append(order_id)
                
            elif updated_order.status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
                for cb in self._on_cancel_callbacks:
                    cb(updated_order)
                completed_ids.append(order_id)

        # Cleanup completed
        for oid in completed_ids:
            del self._monitored_orders[oid]
