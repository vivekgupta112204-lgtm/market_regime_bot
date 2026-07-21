"""Unit tests for Phase 6 — Portfolio Management & Trade Execution Engine."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
import pandas as pd

from config.settings import Settings, ExecutionSettings
from broker.paper_broker import PaperBroker
from execution.order_manager import Order, OrderType, OrderSide, OrderStatus
from execution.execution_engine import ExecutionEngine
from execution.broker_router import BrokerRouter
from portfolio.account import Account
from portfolio.portfolio_state import PortfolioState, OpenPosition
from portfolio.performance_tracker import PerformanceTracker
from portfolio.portfolio_manager import PortfolioManager


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.initial_capital = 100000.0
    s.execution = ExecutionSettings(
        paper_mode=True,
        default_commission=1.0,
        default_slippage_pct=0.001, # 0.1% slippage
        execution_timeout_seconds=5,
        retry_attempts=1,
    )
    s.symbol = "AAPL"
    return s


# ---------------------------------------------------------------------------
# Test Paper Broker
# ---------------------------------------------------------------------------
def test_paper_broker_market_fill(settings):
    broker = PaperBroker(settings.execution, initial_capital=settings.initial_capital)
    broker.connect()
    
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
    
    # 100 price * 0.1% slippage = 0.1 slippage -> fill at 100.1
    submitted = broker.submit_order(order, current_market_price=100.0)
    
    assert submitted.status == OrderStatus.FILLED
    assert submitted.average_fill_price == 100.10
    assert submitted.commission_paid == 1.0
    assert submitted.filled_quantity == 10.0
    
    balances = broker.get_account_balance()
    # Initial 100,000. Bought 10 units at 100.1 = 1001. Plus 1 commission = 1002.
    # Cash remaining = 98998.0
    assert balances["cash"] == 98998.0
    
    positions = broker.get_open_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["quantity"] == 10.0


def test_paper_broker_limit_no_fill(settings):
    broker = PaperBroker(settings.execution, initial_capital=settings.initial_capital)
    broker.connect()
    
    # Buy Limit at 95. Market is 100. Should not fill.
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.LIMIT, limit_price=95.0)
    submitted = broker.submit_order(order, current_market_price=100.0)
    
    assert submitted.status == OrderStatus.PENDING
    assert submitted.filled_quantity == 0.0


# ---------------------------------------------------------------------------
# Test Execution Engine
# ---------------------------------------------------------------------------
def test_execution_engine_routing(settings):
    engine = ExecutionEngine(settings)
    assert isinstance(engine.broker, PaperBroker)
    
    # Simulate a signal from Phase 5 Risk Manager
    mock_risk_signal = {
        "approved": True,
        "signal": "BUY",
        "position_size": 15,
        "entry": 150.0,
        "stop_loss": 140.0,
        "take_profit": 170.0,
    }
    
    result = engine.execute_trade(mock_risk_signal, current_price=150.0)
    
    assert result["status"] == "FILLED"
    assert result["broker"] == "PaperBroker"
    assert result["quantity"] == 15
    assert result["side"] == "BUY"
    assert result["symbol"] == "AAPL"
    
    # Price = 150. Slippage = 0.1% -> 0.15 -> fill at 150.15
    assert result["entry_price"] == 150.15
    assert result["commission"] == 1.0


# ---------------------------------------------------------------------------
# Test Portfolio Management
# ---------------------------------------------------------------------------
def test_performance_tracker():
    tracker = PerformanceTracker()
    tracker.record_trade(150.0)  # Win
    tracker.record_trade(200.0)  # Win
    tracker.record_trade(-100.0) # Loss
    
    metrics = tracker.get_metrics()
    assert metrics["total_trades"] == 3
    assert metrics["win_rate"] == pytest.approx(2/3)
    assert metrics["profit_factor"] == 3.5  # 350 / 100
    assert metrics["total_pnl"] == 250.0
    assert metrics["average_win"] == 175.0
    assert metrics["average_loss"] == 100.0


def test_portfolio_manager_closure():
    acc = Account(initial_capital=10000.0, cash=10000.0)
    state = PortfolioState(account=acc)
    pm = PortfolioManager(state)
    
    # Add a position
    ts = datetime.now(timezone.utc)
    pos = OpenPosition(
        symbol="AAPL", direction="LONG", entry_price=100.0, size=10.0,
        stop_loss=90.0, take_profit=110.0, opened_at=ts, current_price=105.0
    )
    pm.position_manager.add_position(pos)
    pm.cash_manager.allocate_funds(1000.0) # lock $1000 margin
    assert acc.cash == 9000.0
    
    # Close it at 110 (win of $10 per unit, 10 units = $100)
    pm.record_trade_closure("AAPL", exit_price=110.0, quantity=10.0)
    
    assert acc.cash == 10100.0 # 9000 + 1000 margin released + 100 PnL
    assert len(state.positions) == 0
    assert pm.performance_tracker.get_metrics()["total_pnl"] == 100.0
