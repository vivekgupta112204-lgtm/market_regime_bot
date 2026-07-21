"""Execution Engine Orchestrator.

Converts Risk Manager signals into Orders, routes them, and orchestrates
the trade lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from loguru import logger

from config.settings import Settings
from broker.broker_interface import BrokerInterface
from execution.order_manager import Order, OrderType, OrderSide, OrderStatus
from execution.broker_router import BrokerRouter
from execution.order_executor import OrderExecutor
from execution.trade_monitor import TradeMonitor
from execution.sl_tp_manager import SLTPManager


class ExecutionEngine:
    """Central orchestrator for all execution routing and monitoring."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
        # Instantiate sub-components
        self.broker_router = BrokerRouter(settings)
        self.broker: BrokerInterface = self.broker_router.get_broker()
        
        self.executor = OrderExecutor(self.broker, settings.execution)
        self.monitor = TradeMonitor(self.broker)
        self.sltp_manager = SLTPManager(self.executor, self.monitor)

    def execute_trade(self, risk_signal: dict[str, Any], current_price: float = 0.0) -> dict[str, Any]:
        """Convert a Phase 5 Risk Signal into an executed trade.
        
        Args:
            risk_signal: The JSON output from RiskManager.evaluate_trade()
            current_price: Required to simulate execution in PaperBroker.
            
        Returns:
            A JSON-compatible dictionary containing final execution status.
        """
        # Validate input
        if not risk_signal.get("approved", False):
            logger.info("ExecutionEngine: Ignoring rejected signal.")
            return {"status": "REJECTED_BY_RISK"}

        # Construct the parent order
        side = OrderSide.BUY if risk_signal.get("signal") == "BUY" else OrderSide.SELL
        
        parent_order = Order(
            symbol=self.settings.symbol,
            side=side,
            quantity=risk_signal["position_size"],
            order_type=OrderType.MARKET, # Simplified to Market for instant entry
            stop_loss_price=risk_signal.get("stop_loss"),
            take_profit_price=risk_signal.get("take_profit"),
        )
        
        # Execute the order
        logger.info(f"ExecutionEngine: Dispatching {parent_order.side.value} order for {parent_order.quantity} {parent_order.symbol}")
        submitted_order = self.executor.execute_order(parent_order, current_price=current_price)
        
        # Monitor the order (if it didn't instantly fill/reject)
        self.monitor.monitor_order(submitted_order)
        
        # Build the requested JSON output format
        # Note: If it filled instantly (like in paper mode), we return FILLED status immediately.
        # Otherwise it returns SUBMITTED/PENDING.
        
        # Get latest account balances to fulfill the return schema
        balances = self.broker.get_account_balance()
        open_positions = self.broker.get_open_positions()
        
        return {
            "status": submitted_order.status.value,
            "broker": self.broker.__class__.__name__,
            "symbol": submitted_order.symbol,
            "side": submitted_order.side.value,
            "quantity": submitted_order.quantity,
            "entry_price": submitted_order.average_fill_price,
            "stop_loss": submitted_order.stop_loss_price,
            "take_profit": submitted_order.take_profit_price,
            "commission": submitted_order.commission_paid,
            # Slippage is inferred or pulled from config in this summary
            "slippage": self.settings.execution.default_slippage_pct if self.settings.execution.paper_mode else 0.0,
            "portfolio_value": balances.get("equity", 0.0),
            "cash_balance": balances.get("cash", 0.0),
            "open_positions": len(open_positions),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }

    def update(self) -> None:
        """Called periodically to poll broker for order updates."""
        self.monitor.update()
