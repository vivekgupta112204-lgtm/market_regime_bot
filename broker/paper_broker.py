"""Paper Trading Broker.

Simulates order execution with realistic slippage, commissions, and instant/partial fills.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from loguru import logger

from broker.broker_interface import BrokerInterface
from config.settings import ExecutionSettings
from execution.order_manager import Order, OrderStatus, OrderType, OrderSide


class PaperBroker(BrokerInterface):
    """Realistic simulation of a live broker environment."""

    def __init__(self, settings: ExecutionSettings, initial_capital: float = 100000.0) -> None:
        self.settings = settings
        self._connected = False
        
        # Internal state to mock broker backend
        self._cash = initial_capital
        self._equity = initial_capital
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, dict[str, Any]] = {}

    def connect(self) -> bool:
        logger.info("PaperBroker: Connecting to simulated environment...")
        self._connected = True
        return True

    def submit_order(self, order: Order, current_market_price: float = 0.0) -> Order:
        """Simulate order submission and immediate fill logic.
        
        Note: We require current_market_price for simulation purposes.
        In a real broker, the exchange handles this natively.
        """
        if not self._connected:
            order.update_status(OrderStatus.REJECTED)
            return order

        order.broker_id = f"paper_{uuid.uuid4().hex[:8]}"
        order.update_status(OrderStatus.SUBMITTED)
        self._orders[order.broker_id] = order
        
        logger.debug(f"PaperBroker: Received {order.side.value} order for {order.quantity} {order.symbol}")
        
        # Simulate instant execution for Market orders (or stop/limits that cross the spread)
        self._simulate_fill(order, current_market_price)
        
        return order

    def _simulate_fill(self, order: Order, current_price: float) -> None:
        """Simulate execution with slippage and commission."""
        if current_price <= 0:
            logger.warning(f"PaperBroker: Cannot simulate fill without current market price for {order.symbol}")
            return
            
        # Determine if order should fill right now
        should_fill = False
        fill_price = current_price
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
        elif order.order_type == OrderType.LIMIT and order.limit_price:
            if order.side == OrderSide.BUY and current_price <= order.limit_price:
                should_fill = True
                fill_price = order.limit_price # Best effort
            elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                should_fill = True
                fill_price = order.limit_price
        elif order.order_type == OrderType.STOP and order.stop_price:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                should_fill = True
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                should_fill = True
                
        if not should_fill:
            order.update_status(OrderStatus.PENDING)
            return
            
        # Simulate slippage
        # Buy slippage makes price higher, Sell makes it lower
        slippage_amount = fill_price * self.settings.default_slippage_pct
        if order.side == OrderSide.BUY:
            fill_price += slippage_amount
        else:
            fill_price -= slippage_amount
            
        # Update order stats
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission_paid = self.settings.default_commission
        order.update_status(OrderStatus.FILLED)
        
        logger.info(
            f"PaperBroker: FILLED {order.side.value} {order.quantity} {order.symbol} @ {fill_price:.4f} "
            f"(Slip: {self.settings.default_slippage_pct:.2%}, Comm: ${order.commission_paid:.2f})"
        )
        
        # Update internal mock account
        cost = (fill_price * order.quantity) + order.commission_paid
        if order.side == OrderSide.BUY:
            self._cash -= cost
            # Update mock positions
            pos = self._positions.get(order.symbol, {"qty": 0.0, "avg_price": 0.0})
            new_qty = pos["qty"] + order.quantity
            pos["avg_price"] = ((pos["qty"] * pos["avg_price"]) + (order.quantity * fill_price)) / new_qty
            pos["qty"] = new_qty
            self._positions[order.symbol] = pos
        else:
            self._cash += (fill_price * order.quantity) - order.commission_paid
            pos = self._positions.get(order.symbol)
            if pos:
                pos["qty"] -= order.quantity
                if pos["qty"] <= 0:
                    del self._positions[order.symbol]

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                order.update_status(OrderStatus.CANCELED)
                logger.info(f"PaperBroker: Canceled order {order_id}")
                return True
        return False

    def fetch_order_status(self, order: Order) -> Order:
        if order.broker_id and order.broker_id in self._orders:
            return self._orders[order.broker_id]
        return order

    def get_account_balance(self) -> dict[str, float]:
        return {
            "cash": self._cash,
            "equity": self._equity, # In reality, equity fluctuates with open positions
            "margin_used": self._equity - self._cash,
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        result = []
        for sym, data in self._positions.items():
            result.append({
                "symbol": sym,
                "quantity": data["qty"],
                "average_entry_price": data["avg_price"]
            })
        return result

    # Helper method just for the simulator to trigger pending order checks
    def advance_market(self, current_prices: dict[str, float]) -> None:
        """Called each bar to check if pending limit/stop orders should fill."""
        for order in self._orders.values():
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                if order.symbol in current_prices:
                    self._simulate_fill(order, current_prices[order.symbol])
        
        # Update mock equity based on open positions
        self._equity = self._cash
        for sym, data in self._positions.items():
            if sym in current_prices:
                self._equity += data["qty"] * current_prices[sym]
