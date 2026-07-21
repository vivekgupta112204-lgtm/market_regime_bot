"""Unit tests for Phase 4 — Regime-Based Strategy Engine."""

from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd
import pytest

from config.settings import StrategySettings, Settings
from portfolio.exposure import TradeFilter
from portfolio.position import PositionSizer
from signals.signal import SignalType, TradeSignal
from strategy.bull_strategy import BullStrategy
from strategy.bear_strategy import BearStrategy
from strategy.breakout_strategy import BreakoutStrategy
from strategy.strategy_manager import StrategyManager


@pytest.fixture
def strategy_settings() -> StrategySettings:
    return StrategySettings(
        risk_percentage=1.0,
        min_regime_confidence=0.6,
        max_trades_per_day=5,
        session_start_hour=0,
        session_end_hour=23,
    )


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.strategy = StrategySettings(
        risk_percentage=1.0,
        min_regime_confidence=0.6,
        session_start_hour=0,
        session_end_hour=23,
    )
    return s


# ---------------------------------------------------------------------------
# Test Signals
# ---------------------------------------------------------------------------

def test_trade_signal_to_dict():
    sig = TradeSignal(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        regime="Bull",
        strategy="TestStrat",
        signal=SignalType.BUY,
        confidence=0.9,
        entry=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        risk_reward=2.0,
        reason=["Test reason"],
    )
    data = sig.to_dict()
    assert data["signal"] == "BUY"
    assert data["regime"] == "Bull"
    assert data["entry"] == 100.0


def test_create_hold_signal():
    ts = datetime.now(timezone.utc)
    sig = TradeSignal.create_hold(ts, "Bear", "TestStrat", "No setup")
    assert sig.signal == SignalType.HOLD
    assert sig.confidence == 1.0
    assert "No setup" in sig.reason


# ---------------------------------------------------------------------------
# Test Position Sizing
# ---------------------------------------------------------------------------

def test_position_sizer_fixed(strategy_settings):
    sizer = PositionSizer(strategy_settings)
    size = sizer.calculate_size(
        capital=10000.0,
        entry_price=100.0,
        stop_loss=90.0,
        method="fixed_fractional"
    )
    # 10000 capital, 100 price -> 100 units total buying power. 1% risk -> 1 unit.
    assert size == 1.0


def test_position_sizer_atr_risk(strategy_settings):
    sizer = PositionSizer(strategy_settings)
    size = sizer.calculate_size(
        capital=10000.0,
        entry_price=100.0,
        stop_loss=90.0, # risk per unit = 10
        method="atr_risk"
    )
    # 10000 capital, 1% risk = 100 total risk amount.
    # risk per unit = 10. target size = 10 units.
    assert size == 10.0


def test_position_sizer_kelly(strategy_settings):
    sizer = PositionSizer(strategy_settings)
    size = sizer.calculate_size(
        capital=10000.0,
        entry_price=100.0,
        stop_loss=90.0,
        method="kelly",
        win_rate=0.6,
        risk_reward=2.0
    )
    # Kelly = W - ((1-W)/R) = 0.6 - (0.4/2) = 0.4 (40%)
    # Half kelly = 20%. Capped by max risk of 1%.
    # 1% of 10000 = 100 allocated capital. Entry = 100. Size = 1.0
    assert size == 1.0


# ---------------------------------------------------------------------------
# Test Trade Filter
# ---------------------------------------------------------------------------

def test_trade_filter_low_confidence(strategy_settings):
    tf = TradeFilter(strategy_settings)
    sig = TradeSignal(
        timestamp=datetime.now(timezone.utc),
        regime="Bull", strategy="S", signal=SignalType.BUY,
        confidence=0.5, # Below 0.6 min
    )
    assert tf.is_valid(sig) is False


def test_trade_filter_duplicate(strategy_settings):
    tf = TradeFilter(strategy_settings)
    sig = TradeSignal(
        timestamp=datetime.now(timezone.utc),
        regime="Bull", strategy="S", signal=SignalType.BUY,
        confidence=0.9,
    )
    assert tf.is_valid(sig) is True
    tf.record_execution(sig)
    
    # Second buy should be rejected as duplicate
    sig2 = TradeSignal(
        timestamp=datetime.now(timezone.utc),
        regime="Bull", strategy="S", signal=SignalType.BUY,
        confidence=0.9,
    )
    assert tf.is_valid(sig2) is False


# ---------------------------------------------------------------------------
# Test Strategy Logic
# ---------------------------------------------------------------------------

def test_bull_strategy_entry(strategy_settings):
    strat = BullStrategy(strategy_settings)
    data = pd.Series({
        "timestamp": datetime.now(timezone.utc),
        "close": 100.0,
        "ema_20": 105.0,
        "ema_50": 100.0,
        "macd_histogram": 0.5,
        "adx": 30.0,
        "atr": 2.0,
    })
    
    sig = strat.generate_signal(data, "Bull Market", 0.9)
    assert sig.signal == SignalType.BUY
    assert sig.stop_loss == 100.0 - (2.0 * strategy_settings.atr_multiplier_stop_loss)


def test_breakout_strategy(strategy_settings):
    strat = BreakoutStrategy(strategy_settings)
    data = pd.Series({
        "timestamp": datetime.now(timezone.utc),
        "close": 105.0,
        "donchian_upper": 105.0,
        "donchian_lower": 90.0,
        "volume_change": 0.6,
        "atr": 2.0,
    })
    
    sig = strat.generate_signal(data, "Low Volatility", 0.8)
    assert sig.signal == SignalType.BUY


# ---------------------------------------------------------------------------
# Test Strategy Manager
# ---------------------------------------------------------------------------

def test_strategy_manager_routing(settings):
    manager = StrategyManager(settings)
    
    data = pd.Series({
        "timestamp": datetime.now(timezone.utc),
        "close": 100.0,
        "ema_20": 90.0,
        "ema_50": 100.0,
        "macd_histogram": -0.5,
        "rsi": 40.0,
        "atr": 2.0,
    })
    
    # Pass "Bear Market" regime
    result = manager.generate_trade_signal(data, regime="Bear Market", confidence=0.85)
    
    assert result["strategy"] == "BearTrendShorting"
    assert result["signal"] == "SHORT"
    assert result["confidence"] == 0.85
    assert result["entry"] == 100.0
