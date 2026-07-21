"""Unit tests for Phase 7 — Backtesting, Optimization & Performance Analytics."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings, BacktestSettings


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def settings() -> Settings:
    """Default test settings."""
    return Settings(initial_capital=100_000.0, symbol="TEST")


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 200 bars of synthetic OHLCV data with a gentle uptrend."""
    np.random.seed(42)
    n = 200
    dates = [datetime(2023, 1, 2, tzinfo=timezone.utc) + timedelta(days=i) for i in range(n)]

    close = [100.0]
    for i in range(1, n):
        change = np.random.normal(0.0005, 0.015)
        close.append(close[-1] * (1 + change))

    close = np.array(close)
    high = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(100_000, 1_000_000, n).astype(float)

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def equity_series() -> pd.Series:
    """Synthetic equity curve for analytics tests."""
    np.random.seed(123)
    n = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    values = [100_000.0]
    for _ in range(1, n):
        change = np.random.normal(0.0004, 0.01)
        values.append(values[-1] * (1 + change))
    return pd.Series(values, index=dates, name="portfolio_value")


@pytest.fixture
def sample_trades() -> list[dict]:
    """Synthetic closed trade list for analytics tests."""
    return [
        {"pnl": 500.0, "pnl_pct": 0.05, "bars_held": 5, "symbol": "TEST", "direction": "LONG"},
        {"pnl": -200.0, "pnl_pct": -0.02, "bars_held": 3, "symbol": "TEST", "direction": "LONG"},
        {"pnl": 300.0, "pnl_pct": 0.03, "bars_held": 7, "symbol": "TEST", "direction": "LONG"},
        {"pnl": 150.0, "pnl_pct": 0.015, "bars_held": 2, "symbol": "TEST", "direction": "SHORT"},
        {"pnl": -100.0, "pnl_pct": -0.01, "bars_held": 4, "symbol": "TEST", "direction": "LONG"},
        {"pnl": 400.0, "pnl_pct": 0.04, "bars_held": 6, "symbol": "TEST", "direction": "LONG"},
        {"pnl": -50.0, "pnl_pct": -0.005, "bars_held": 1, "symbol": "TEST", "direction": "SHORT"},
        {"pnl": 250.0, "pnl_pct": 0.025, "bars_held": 8, "symbol": "TEST", "direction": "LONG"},
    ]


# ═══════════════════════════════════════════════════════════════════════
# Commission Model
# ═══════════════════════════════════════════════════════════════════════

class TestCommissionModel:

    def test_flat_commission(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        model = CommissionModel(commission_type=CommissionType.FLAT, flat_fee=5.0)
        cost = model.calculate(100, 50.0)
        assert cost == 5.0

    def test_percentage_commission(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        model = CommissionModel(commission_type=CommissionType.PERCENTAGE, percentage_rate=0.001)
        cost = model.calculate(100, 50.0)
        # notional = 5000, commission = 5000 * 0.001 = 5.0
        assert cost == 5.0

    def test_tiered_commission(self):
        from backtesting.commission_model import CommissionModel, CommissionType, CommissionTier
        tiers = [
            CommissionTier(max_notional=1000, rate=0.002),
            CommissionTier(max_notional=5000, rate=0.001),
            CommissionTier(max_notional=float("inf"), rate=0.0005),
        ]
        model = CommissionModel(commission_type=CommissionType.TIERED, tiers=tiers)
        # notional = 100 * 50 = 5000
        # first 1000 @ 0.002 = 2.0, next 4000 @ 0.001 = 4.0 => total = 6.0
        cost = model.calculate(100, 50.0)
        assert cost == pytest.approx(6.0, abs=0.01)

    def test_exchange_fee(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        model = CommissionModel(
            commission_type=CommissionType.FLAT, flat_fee=1.0,
            exchange_fee_rate=0.0001,
        )
        # notional = 10 * 100 = 1000, exchange = 0.1
        cost = model.calculate(10, 100.0)
        assert cost == pytest.approx(1.1, abs=0.01)

    def test_borrow_fee(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        model = CommissionModel(
            commission_type=CommissionType.FLAT, flat_fee=1.0,
            borrow_fee_annual=0.05,
        )
        cost = model.calculate(10, 100.0, is_short=True, holding_days=30)
        # borrow = 1000 * (0.05/365) * 30 ≈ 4.1096
        assert cost > 1.0
        assert cost == pytest.approx(5.1096, abs=0.01)

    def test_min_commission(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        model = CommissionModel(
            commission_type=CommissionType.PERCENTAGE,
            percentage_rate=0.001,
            min_commission=10.0,
        )
        # notional = 1 * 10 = 10, pct comm = 0.01, min = 10
        cost = model.calculate(1, 10.0)
        assert cost == 10.0


# ═══════════════════════════════════════════════════════════════════════
# Slippage Model
# ═══════════════════════════════════════════════════════════════════════

class TestSlippageModel:

    def test_fixed_slippage_buy(self):
        from backtesting.slippage_model import SlippageModel, SlippageType
        model = SlippageModel(slippage_type=SlippageType.FIXED, fixed_pct=0.001)
        price = model.calculate(100.0, 10, is_buy=True)
        # 100 * (1 + 0.001) = 100.1
        assert price == pytest.approx(100.1, abs=0.001)

    def test_fixed_slippage_sell(self):
        from backtesting.slippage_model import SlippageModel, SlippageType
        model = SlippageModel(slippage_type=SlippageType.FIXED, fixed_pct=0.001)
        price = model.calculate(100.0, 10, is_buy=False)
        # 100 * (1 - 0.001) = 99.9
        assert price == pytest.approx(99.9, abs=0.001)

    def test_volume_based_slippage(self):
        from backtesting.slippage_model import SlippageModel, SlippageType
        model = SlippageModel(
            slippage_type=SlippageType.VOLUME_BASED,
            fixed_pct=0.0005,
            volume_impact_factor=0.1,
        )
        # participation = 100 / 10000 = 0.01
        # slippage = 0.0005 + 0.01 * 0.1 = 0.0015
        price = model.calculate(100.0, 100, bar_volume=10000, is_buy=True)
        assert price == pytest.approx(100.15, abs=0.01)

    def test_slippage_cost(self):
        from backtesting.slippage_model import SlippageModel, SlippageType
        model = SlippageModel(slippage_type=SlippageType.FIXED, fixed_pct=0.001)
        cost = model.calculate_cost(100.0, 10, is_buy=True)
        # cost = |100.1 - 100| * 10 = 1.0
        assert cost == pytest.approx(1.0, abs=0.01)

    def test_max_slippage_cap(self):
        from backtesting.slippage_model import SlippageModel, SlippageType
        model = SlippageModel(
            slippage_type=SlippageType.VOLUME_BASED,
            fixed_pct=0.05,
            volume_impact_factor=1.0,
            max_slippage_pct=0.01,
        )
        # With huge participation rate, slippage would exceed max
        price = model.calculate(100.0, 5000, bar_volume=100, is_buy=True)
        assert price <= 101.01  # max 1% slip


# ═══════════════════════════════════════════════════════════════════════
# Trade Simulator
# ═══════════════════════════════════════════════════════════════════════

class TestTradeSimulator:

    def test_market_order_fills(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        from backtesting.slippage_model import SlippageModel
        from backtesting.trade_simulator import (
            TradeSimulator, SimulatedOrder, OrderSide, SimOrderType, FillStatus,
        )

        comm = CommissionModel(commission_type=CommissionType.FLAT, flat_fee=1.0)
        slip = SlippageModel(fixed_pct=0.001)
        sim = TradeSimulator(comm, slip)

        order = SimulatedOrder(
            symbol="TEST", side=OrderSide.BUY,
            order_type=SimOrderType.MARKET, quantity=10,
        )
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)
        fill = sim.execute(order, 100.0, 102.0, 99.0, 101.0, 100000, ts)

        assert fill.status == FillStatus.FILLED
        assert fill.quantity == 10
        assert fill.price == pytest.approx(100.1, abs=0.01)  # 100 * 1.001
        assert fill.commission == 1.0

    def test_limit_order_not_triggered(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        from backtesting.slippage_model import SlippageModel
        from backtesting.trade_simulator import (
            TradeSimulator, SimulatedOrder, OrderSide, SimOrderType, FillStatus,
        )

        comm = CommissionModel(commission_type=CommissionType.FLAT, flat_fee=1.0)
        slip = SlippageModel(fixed_pct=0.0)
        sim = TradeSimulator(comm, slip)

        # Buy limit at 95, bar low is 98 → not triggered
        order = SimulatedOrder(
            symbol="TEST", side=OrderSide.BUY,
            order_type=SimOrderType.LIMIT, quantity=10, limit_price=95.0,
        )
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)
        fill = sim.execute(order, 100.0, 102.0, 98.0, 101.0, 100000, ts)

        assert fill.status == FillStatus.PENDING
        assert fill.quantity == 0.0

    def test_limit_order_triggered(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        from backtesting.slippage_model import SlippageModel
        from backtesting.trade_simulator import (
            TradeSimulator, SimulatedOrder, OrderSide, SimOrderType, FillStatus,
        )

        comm = CommissionModel(commission_type=CommissionType.FLAT, flat_fee=1.0)
        slip = SlippageModel(fixed_pct=0.0)
        sim = TradeSimulator(comm, slip)

        # Buy limit at 99, bar low is 98 → triggered
        order = SimulatedOrder(
            symbol="TEST", side=OrderSide.BUY,
            order_type=SimOrderType.LIMIT, quantity=10, limit_price=99.0,
        )
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)
        fill = sim.execute(order, 100.0, 102.0, 98.0, 101.0, 100000, ts)

        assert fill.status == FillStatus.FILLED
        assert fill.price == pytest.approx(99.0, abs=0.01)

    def test_stop_order_triggered(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        from backtesting.slippage_model import SlippageModel
        from backtesting.trade_simulator import (
            TradeSimulator, SimulatedOrder, OrderSide, SimOrderType, FillStatus,
        )

        comm = CommissionModel(commission_type=CommissionType.FLAT, flat_fee=1.0)
        slip = SlippageModel(fixed_pct=0.0)
        sim = TradeSimulator(comm, slip)

        # Sell stop at 95, bar low is 94 → triggered
        order = SimulatedOrder(
            symbol="TEST", side=OrderSide.SELL,
            order_type=SimOrderType.STOP, quantity=10, limit_price=95.0,
        )
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)
        fill = sim.execute(order, 100.0, 102.0, 94.0, 96.0, 100000, ts)

        assert fill.status == FillStatus.FILLED

    def test_order_rejection(self):
        from backtesting.commission_model import CommissionModel, CommissionType
        from backtesting.slippage_model import SlippageModel
        from backtesting.trade_simulator import (
            TradeSimulator, SimulatedOrder, OrderSide, SimOrderType, FillStatus,
        )

        comm = CommissionModel(commission_type=CommissionType.FLAT, flat_fee=1.0)
        slip = SlippageModel(fixed_pct=0.001)
        sim = TradeSimulator(comm, slip, min_order_size=1.0)

        order = SimulatedOrder(
            symbol="TEST", side=OrderSide.BUY,
            order_type=SimOrderType.MARKET, quantity=0.001,
        )
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)
        fill = sim.execute(order, 100.0, 102.0, 99.0, 101.0, 100000, ts)

        assert fill.status == FillStatus.REJECTED


# ═══════════════════════════════════════════════════════════════════════
# Portfolio Simulator
# ═══════════════════════════════════════════════════════════════════════

class TestPortfolioSimulator:

    def test_initial_state(self):
        from backtesting.simulator import PortfolioSimulator
        sim = PortfolioSimulator(100_000.0, "TEST")
        assert sim.cash == 100_000.0
        assert sim.initial_capital == 100_000.0
        assert len(sim.positions) == 0
        assert len(sim.closed_trades) == 0

    def test_open_and_close_position(self):
        from backtesting.simulator import PortfolioSimulator
        from backtesting.trade_simulator import SimulatedFill, OrderSide, FillStatus

        sim = PortfolioSimulator(100_000.0, "TEST")
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)

        # Open a long position
        fill = SimulatedFill(
            symbol="TEST", side=OrderSide.BUY, quantity=10, price=100.0,
            commission=1.0, slippage_cost=0.5, timestamp=ts, status=FillStatus.FILLED,
        )
        sim.process_fill(fill, stop_loss=95.0, take_profit=110.0)

        assert len(sim.positions) == 1
        assert sim.positions["TEST"].quantity == 10
        assert sim.cash < 100_000.0

        # Close the position
        close_fill = SimulatedFill(
            symbol="TEST", side=OrderSide.SELL, quantity=10, price=105.0,
            commission=1.0, slippage_cost=0.5, timestamp=ts, status=FillStatus.FILLED,
        )
        sim.process_fill(close_fill)

        assert len(sim.positions) == 0
        assert len(sim.closed_trades) == 1
        assert sim.closed_trades[0].pnl > 0  # profitable trade

    def test_equity_snapshot(self):
        from backtesting.simulator import PortfolioSimulator
        sim = PortfolioSimulator(100_000.0, "TEST")
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)

        snap = sim.snapshot(ts, 100.0)
        assert snap.portfolio_value == 100_000.0
        assert snap.cash == 100_000.0
        assert snap.open_positions == 0

    def test_stop_loss_check(self):
        from backtesting.simulator import PortfolioSimulator
        from backtesting.trade_simulator import SimulatedFill, OrderSide, FillStatus

        sim = PortfolioSimulator(100_000.0, "TEST")
        ts = datetime(2023, 6, 1, tzinfo=timezone.utc)

        fill = SimulatedFill(
            symbol="TEST", side=OrderSide.BUY, quantity=10, price=100.0,
            commission=0.0, slippage_cost=0.0, timestamp=ts, status=FillStatus.FILLED,
        )
        sim.process_fill(fill, stop_loss=95.0, take_profit=110.0)

        # Bar low = 94, should trigger SL
        closed = sim.check_stops("TEST", 101.0, 94.0, 96.0, ts)
        assert closed is not None
        assert closed.exit_price == 95.0
        assert len(sim.positions) == 0


# ═══════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════

class TestStatistics:

    def test_total_return(self, equity_series):
        from analytics.statistics import total_return
        ret = total_return(equity_series)
        assert isinstance(ret, float)

    def test_sharpe_ratio(self, equity_series):
        from analytics.statistics import sharpe_ratio
        sr = sharpe_ratio(equity_series)
        assert isinstance(sr, float)

    def test_sortino_ratio(self, equity_series):
        from analytics.statistics import sortino_ratio
        sor = sortino_ratio(equity_series)
        assert isinstance(sor, float)

    def test_cagr(self, equity_series):
        from analytics.statistics import cagr
        c = cagr(equity_series)
        assert isinstance(c, float)

    def test_profit_factor(self, sample_trades):
        from analytics.statistics import profit_factor
        pf = profit_factor(sample_trades)
        # gross profit = 1600, gross loss = 350
        assert pf == pytest.approx(1600.0 / 350.0, abs=0.1)

    def test_win_rate(self, sample_trades):
        from analytics.statistics import win_rate
        wr = win_rate(sample_trades)
        # 5 winners out of 8 = 62.5%
        assert wr == pytest.approx(62.5, abs=0.1)

    def test_compute_all(self, equity_series, sample_trades):
        from analytics.statistics import compute_all
        result = compute_all(equity_series, sample_trades)
        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "total_trades" in result
        assert result["total_trades"] == 8


# ═══════════════════════════════════════════════════════════════════════
# Risk Metrics
# ═══════════════════════════════════════════════════════════════════════

class TestRiskMetrics:

    def test_var_historical(self, equity_series):
        from analytics.risk_metrics import value_at_risk
        var = value_at_risk(equity_series, 0.95, "historical")
        assert var >= 0

    def test_var_parametric(self, equity_series):
        from analytics.risk_metrics import value_at_risk
        var = value_at_risk(equity_series, 0.95, "parametric")
        assert var >= 0

    def test_cvar(self, equity_series):
        from analytics.risk_metrics import conditional_var
        cvar = conditional_var(equity_series, 0.95)
        assert cvar >= 0

    def test_volatility(self, equity_series):
        from analytics.risk_metrics import portfolio_volatility
        vol = portfolio_volatility(equity_series)
        assert vol > 0

    def test_ulcer_index(self, equity_series):
        from analytics.risk_metrics import ulcer_index
        ui = ulcer_index(equity_series)
        assert ui >= 0

    def test_compute_all(self, equity_series):
        from analytics.risk_metrics import compute_all
        result = compute_all(equity_series)
        assert "volatility" in result
        assert "var_95" in result
        assert "ulcer_index" in result

    def test_beta_alpha_with_benchmark(self, equity_series):
        from analytics.risk_metrics import beta, alpha
        # Use a shifted version as benchmark
        bench = equity_series * 1.05
        b = beta(equity_series, bench)
        a = alpha(equity_series, bench)
        assert isinstance(b, float)
        assert isinstance(a, float)


# ═══════════════════════════════════════════════════════════════════════
# Drawdown Analyzer
# ═══════════════════════════════════════════════════════════════════════

class TestDrawdownAnalyzer:

    def test_max_drawdown(self, equity_series):
        from analytics.drawdown import DrawdownAnalyzer
        analyzer = DrawdownAnalyzer(equity_series)
        md = analyzer.max_drawdown
        assert md >= 0

    def test_drawdown_events(self, equity_series):
        from analytics.drawdown import DrawdownAnalyzer
        analyzer = DrawdownAnalyzer(equity_series)
        events = analyzer.events
        assert isinstance(events, list)

    def test_drawdown_series(self, equity_series):
        from analytics.drawdown import DrawdownAnalyzer
        analyzer = DrawdownAnalyzer(equity_series)
        dd = analyzer.drawdown_series
        assert len(dd) == len(equity_series)
        assert (dd >= 0).all()

    def test_known_drawdown(self):
        """Test drawdown on a known sequence: 100 → 110 → 88 → 100."""
        from analytics.drawdown import DrawdownAnalyzer
        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        values = [100.0, 110.0, 88.0, 100.0]
        series = pd.Series(values, index=dates)
        analyzer = DrawdownAnalyzer(series)
        # Max drawdown = (110 - 88) / 110 * 100 = 20%
        assert analyzer.max_drawdown == pytest.approx(20.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# Trade Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestTradeAnalysis:

    def test_winning_trades(self, sample_trades):
        from analytics.trade_analysis import winning_trades
        stats = winning_trades(sample_trades)
        assert stats.count == 5

    def test_losing_trades(self, sample_trades):
        from analytics.trade_analysis import losing_trades
        stats = losing_trades(sample_trades)
        assert stats.count == 3

    def test_best_trade(self, sample_trades):
        from analytics.trade_analysis import best_trade
        best = best_trade(sample_trades)
        assert best["pnl"] == 500.0

    def test_worst_trade(self, sample_trades):
        from analytics.trade_analysis import worst_trade
        worst = worst_trade(sample_trades)
        assert worst["pnl"] == -200.0

    def test_win_streak(self, sample_trades):
        from analytics.trade_analysis import longest_win_streak
        # Trades: +, -, +, +, -, +, -, + → longest win streak = 2
        streak = longest_win_streak(sample_trades)
        assert streak == 2

    def test_loss_streak(self, sample_trades):
        from analytics.trade_analysis import longest_loss_streak
        streak = longest_loss_streak(sample_trades)
        assert streak == 1

    def test_compute_all(self, sample_trades):
        from analytics.trade_analysis import compute_all
        result = compute_all(sample_trades)
        assert "total_trades" in result
        assert "winning" in result
        assert "longest_win_streak" in result
        assert "duration_distribution" in result


# ═══════════════════════════════════════════════════════════════════════
# Performance Analyzer
# ═══════════════════════════════════════════════════════════════════════

class TestPerformanceAnalyzer:

    def test_end_to_end(self, equity_series, sample_trades):
        from analytics.performance import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(equity_series, sample_trades)
        result = analyzer.compute_all()

        assert "performance" in result
        assert "risk" in result
        assert "drawdown" in result
        assert "trades" in result

    def test_summary(self, equity_series, sample_trades):
        from analytics.performance import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(equity_series, sample_trades)
        summary = analyzer.summary()

        assert "initial_capital" in summary
        assert "final_capital" in summary
        assert "sharpe_ratio" in summary
        assert "total_trades" in summary


# ═══════════════════════════════════════════════════════════════════════
# Backtest Engine
# ═══════════════════════════════════════════════════════════════════════

class TestBacktestEngine:

    def test_full_backtest(self, settings, synthetic_ohlcv):
        from backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(settings, initial_capital=100_000.0)
        result = engine.run(synthetic_ohlcv)

        assert "initial_capital" in result
        assert "final_capital" in result
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "total_trades" in result
        assert "equity_series" in result
        assert "trades" in result
        assert result["initial_capital"] == 100_000.0
        assert isinstance(result["equity_series"], pd.Series)

    def test_run_backtest_convenience(self, settings, synthetic_ohlcv):
        from backtesting.backtest_engine import run_backtest

        result = run_backtest(
            settings=settings,
            df=synthetic_ohlcv,
            initial_capital=100_000.0,
        )

        assert "total_return" in result
        assert "analytics" in result


# ═══════════════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════════════

class TestReportGenerator:

    def test_html_report(self, settings, synthetic_ohlcv):
        from backtesting.backtest_engine import BacktestEngine
        from reports.report_generator import ReportGenerator

        engine = BacktestEngine(settings, initial_capital=100_000.0)
        result = engine.run(synthetic_ohlcv)

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(result, symbol="TEST", output_dir=tmpdir)
            path = gen.generate_html()
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "Backtest Report" in content

    def test_json_report(self, settings, synthetic_ohlcv):
        from backtesting.backtest_engine import BacktestEngine
        from reports.report_generator import ReportGenerator

        engine = BacktestEngine(settings, initial_capital=100_000.0)
        result = engine.run(synthetic_ohlcv)

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(result, symbol="TEST", output_dir=tmpdir)
            path = gen.generate_json()
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "total_return" in data

    def test_csv_report(self, settings, synthetic_ohlcv):
        from backtesting.backtest_engine import BacktestEngine
        from reports.report_generator import ReportGenerator

        engine = BacktestEngine(settings, initial_capital=100_000.0)
        result = engine.run(synthetic_ohlcv)

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(result, symbol="TEST", output_dir=tmpdir)
            path = gen.generate_csv()
            assert path.exists()


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

class TestBacktestSettings:

    def test_default_settings(self):
        bs = BacktestSettings()
        assert bs.commission_rate == 0.001
        assert bs.slippage_pct == 0.0005
        assert bs.benchmark_symbol == "SPY"

    def test_settings_integration(self):
        s = Settings()
        assert hasattr(s, "backtest")
        assert s.backtest.commission_rate == 0.001
