"""SL / TP Manager.

Handles submission of child orders (Stop Loss and Take Profit) for a filled parent order.
"""

from __future__ import annotations

from loguru import logger

from execution.order_manager import Order, OrderType, OrderSide
from execution.order_executor import OrderExecutor
from execution.trade_monitor import TradeMonitor


class SLTPManager:
    """Manages child stop-loss and take-profit orders."""

    def __init__(self, executor: OrderExecutor, monitor: TradeMonitor) -> None:
        self.executor = executor
        self.monitor = monitor
        
        # We hook into the monitor to automatically place SL/TP when a parent fills
        self.monitor.register_on_fill(self._on_parent_filled)

    def _on_parent_filled(self, parent_order: Order) -> None:
        """Triggered when any order is filled."""
        # Check if the parent had attached SL/TP prices
        if parent_order.stop_loss_price is None and parent_order.take_profit_price is None:
            return
            
        logger.info(f"SLTPManager: Placing child orders for filled parent {parent_order.id}")
        
        # The closing side is opposite to the parent side
        exit_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
        qty = parent_order.filled_quantity

        if parent_order.stop_loss_price:
            sl_order = Order(
                symbol=parent_order.symbol,
                side=exit_side,
                quantity=qty,
                order_type=OrderType.STOP,
                stop_price=parent_order.stop_loss_price
            )
            submitted_sl = self.executor.execute_order(sl_order)
            self.monitor.monitor_order(submitted_sl)

        if parent_order.take_profit_price:
            tp_order = Order(
                symbol=parent_order.symbol,
                side=exit_side,
                quantity=qty,
                order_type=OrderType.LIMIT,
                limit_price=parent_order.take_profit_price
            )
            submitted_tp = self.executor.execute_order(tp_order)
            self.monitor.monitor_order(submitted_tp)
            
        # Note: True OCO (One-Cancels-Other) requires broker support or active monitoring
        # to cancel the other leg when one fills. For this abstraction, we assume independent legs
        # or that the broker links them natively if sent as a BRACKET order initially.
