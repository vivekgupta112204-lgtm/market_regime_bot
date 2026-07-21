"""Order Executor.

Handles the dispatch of orders to the broker, managing retries and timeouts.
"""

from __future__ import annotations

import time
from loguru import logger

from broker.broker_interface import BrokerInterface
from execution.order_manager import Order, OrderStatus
from config.settings import ExecutionSettings


class OrderExecutor:
    """Dispatches orders to the broker with retry logic."""

    def __init__(self, broker: BrokerInterface, settings: ExecutionSettings) -> None:
        self.broker = broker
        self.settings = settings

    def execute_order(self, order: Order, current_price: float = 0.0) -> Order:
        """Attempt to send an order to the broker.
        
        Args:
            order: The Order to send.
            current_price: Current market price (used primarily for PaperBroker).
            
        Returns:
            The submitted Order (potentially with filled status or rejected).
        """
        attempts = 0
        max_attempts = self.settings.retry_attempts + 1
        
        while attempts < max_attempts:
            try:
                # We pass current_price specifically for PaperBroker's synchronous fills
                if hasattr(self.broker, "submit_order") and "current_market_price" in self.broker.submit_order.__code__.co_varnames:
                    return self.broker.submit_order(order, current_market_price=current_price) # type: ignore
                else:
                    return self.broker.submit_order(order)
                    
            except Exception as e:
                attempts += 1
                logger.error(f"OrderExecutor: Failed to submit order {order.id}. Attempt {attempts}/{max_attempts}. Error: {e}")
                if attempts < max_attempts:
                    time.sleep(1.0) # Simple backoff
                    
        # If we exhausted attempts
        order.update_status(OrderStatus.REJECTED)
        return order
