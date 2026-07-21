"""Unit tests for Phase 5 — Advanced Risk Management Engine."""

from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd
import pytest

from config.settings import Settings, RiskSettings, StrategySettings
from portfolio.account import Account
from portfolio.portfolio_state import PortfolioState, OpenPosition
from signals.signal import SignalType, TradeSignal

from risk.risk_manager import RiskManager
from risk.position_sizing import AdvancedPositionSizer
from risk.stop_loss import StopLossEngine
from risk.take_profit import TakeProfitEngine
from risk.drawdown_guard import DrawdownGuard
from risk.daily_loss_guard import DailyLossGuard
from risk.exposure_manager import ExposureManager
from risk.volatility_filter import VolatilityFilter
from risk.correlation_filter import CorrelationFilter


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.initial_capital = 100000.0
    s.strategy = StrategySettings(
        risk_percentage=1.0,
        leverage=1.0,
        max_position_size=100000.0,
        min_lot_size=1.0,
    )
    s.risk = RiskSettings(
        max_daily_loss_fiat=1000.0,
        max_drawdown_pct=0.10,
        max_open_positions=3,
        max_portfolio_exposure_pct=0.50,
        correlation_threshold=0.75,
    )
    return s


@pytest.fixture
def portfolio(settings) -> PortfolioState:
    acc = Account(
        initial_capital=settings.initial_capital,
        cash=settings.initial_capital,
    )
    return PortfolioState(account=acc)


# ---------------------------------------------------------------------------
# Test Drawdown Guard
# ---------------------------------------------------------------------------
def test_drawdown_guard_safe(settings, portfolio):
    guard = DrawdownGuard(settings.risk)
    assert guard.is_safe(portfolio) is True


def test_drawdown_guard_breached(settings, portfolio):
    guard = DrawdownGuard(settings.risk)
    # Simulate a 15% drawdown
    portfolio.account.high_water_mark = 100000.0
    # Add a massive losing position
    portfolio.positions["BAD"] = OpenPosition(
        symbol="BAD", direction="LONG", entry_price=100.0, size=500.0,
        stop_loss=0.0, take_profit=200.0, opened_at=datetime.now(timezone.utc),
        current_price=70.0 # $15,000 loss
    )
    assert guard.is_safe(portfolio) is False


# ---------------------------------------------------------------------------
# Test Daily Loss Guard
# ---------------------------------------------------------------------------
def test_daily_loss_guard_safe(settings, portfolio):
    guard = DailyLossGuard(settings.risk)
    assert guard.is_safe(portfolio) is True


def test_daily_loss_guard_breached(settings, portfolio):
    guard = DailyLossGuard(settings.risk)
    guard.record_closed_trade(-1500.0) # Hit the $1000 max daily loss
    assert guard.is_safe(portfolio) is False


# ---------------------------------------------------------------------------
# Test Exposure Manager
# ---------------------------------------------------------------------------
def test_exposure_manager_max_positions(settings, portfolio):
    manager = ExposureManager(settings.risk)
    ts = datetime.now(timezone.utc)
    for i in range(3):
        portfolio.positions[f"SYM{i}"] = OpenPosition(
            symbol=f"SYM{i}", direction="LONG", entry_price=10.0, size=1.0,
            stop_loss=9.0, take_profit=12.0, opened_at=ts, current_price=10.0
        )
    # 4th position should be rejected
    assert manager.is_safe("NEW_SYM", 1000.0, portfolio) is False


def test_exposure_manager_max_exposure(settings, portfolio):
    manager = ExposureManager(settings.risk)
    # Max exposure is 50% of 100,000 = 50,000.
    assert manager.is_safe("SYM1", 40000.0, portfolio) is True
    assert manager.is_safe("SYM1", 60000.0, portfolio) is False


# ---------------------------------------------------------------------------
# Test Advanced Position Sizer
# ---------------------------------------------------------------------------
def test_position_sizer_kelly(settings, portfolio):
    sizer = AdvancedPositionSizer(settings.strategy, settings.risk)
    size = sizer.calculate_size(
        portfolio=portfolio,
        entry_price=100.0,
        stop_loss=90.0,
        method="kelly",
        win_rate=0.6,
        risk_reward=2.0
    )
    # W=0.6, R=2.0 -> Kelly=0.4. Half-Kelly=0.2. Capped at Risk%=0.01
    # 100,000 * 0.01 = 1000 target capital. Entry=100 -> Size=10
    assert size == 10.0


def test_position_sizer_max_dollar_risk(settings, portfolio):
    sizer = AdvancedPositionSizer(settings.strategy, settings.risk)
    size = sizer.calculate_size(
        portfolio=portfolio,
        entry_price=100.0,
        stop_loss=90.0, # Risk/unit = 10
        method="max_dollar_risk"
    )
    # Max daily loss is 1000. Half is 500.
    # 500 / 10 = 50 units.
    assert size == 50.0


# ---------------------------------------------------------------------------
# Test Risk Manager Orchestration
# ---------------------------------------------------------------------------
def test_risk_manager_approval(settings, portfolio):
    rm = RiskManager(settings)
    
    ts = datetime.now(timezone.utc)
    sig = TradeSignal(
        timestamp=ts, regime="Bull", strategy="Test", signal=SignalType.BUY,
        confidence=0.9, entry=100.0, stop_loss=0.0, take_profit=0.0,
        risk_reward=0.0, reason=[]
    )
    data = pd.Series({
        "timestamp": ts, "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0,
        "atr": 2.0, "prev_close": 100.0
    })
    
    result = rm.evaluate_trade(sig, data, portfolio)
    assert result["approved"] is True
    # Stop loss calculated via ATR (multiplier=2.0, atr=2.0 -> 4.0 dist)
    assert result["stop_loss"] == 96.0
    assert result["position_size"] > 0
    assert "portfolio_exposure" in result
