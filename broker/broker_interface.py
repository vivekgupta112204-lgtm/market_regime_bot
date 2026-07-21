"""Broker Interface definitions.

Provides a unified interface for paper and live brokers.
"""

from __future__ import annotations

import abc
from typing import Any

from execution.order_manager import Order


class BrokerInterface(abc.ABC):
    """Abstract base class for all broker integrations."""

    @abc.abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish connection with the broker."""
        pass

    @abc.abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker.
        
        Args:
            order: The internal Order object.
            
        Returns:
            The Order object updated with the broker_id and initial status.
        """
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel a pending order.
        
        Args:
            order_id: The internal or broker order ID.
            
        Returns:
            True if cancellation was requested successfully.
        """
        pass

    @abc.abstractmethod
    def fetch_order_status(self, order: Order) -> Order:
        """Query the broker for the latest status of an order.
        
        Args:
            order: The Order object to update.
            
        Returns:
            The updated Order object.
        """
        pass

    @abc.abstractmethod
    def get_account_balance(self) -> dict[str, float]:
        """Fetch current account balances from the broker.
        
        Returns:
            Dictionary containing 'cash', 'equity', 'margin_used', etc.
        """
        pass

    @abc.abstractmethod
    def get_open_positions(self) -> list[dict[str, Any]]:
        """Fetch all currently open positions from the broker.
        
        Returns:
            List of dictionaries containing position details (symbol, size, avg_price).
        """
        pass
