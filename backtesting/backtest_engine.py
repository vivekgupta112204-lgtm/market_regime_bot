"""Event-driven backtesting engine.

Replays historical candles bar-by-bar through the full trading pipeline:
Feature Engineering → Regime Detection → Strategy → Risk Manager → Trade
Simulation → Portfolio Tracking → Analytics.

Exposes ``run_backtest()`` as the top-level public API.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import Settings
from features.feature_engineering import compute_all_features
from signals.signal import SignalType, TradeSignal
from backtesting.commission_model import CommissionModel, CommissionType
from backtesting.slippage_model import SlippageModel
from backtesting.trade_simulator import (
    TradeSimulator,
    SimulatedOrder,
    OrderSide,
    SimOrderType,
    FillStatus,
)
from backtesting.simulator import PortfolioSimulator, ClosedTrade
from analytics.performance import PerformanceAnalyzer


class BacktestEngine:
    """Event-driven backtesting engine.

    Processes historical OHLCV data bar-by-bar, integrating with the
    existing strategy, risk, and regime-detection modules.

    Args:
        settings: Application settings.
        commission_model: Commission/fee calculator.
        slippage_model: Slippage calculator.
        initial_capital: Starting portfolio value.
    """

    def __init__(
        self,
        settings: Settings,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        initial_capital: float | None = None,
    ) -> None:
        self._settings = settings
        self._initial_capital = initial_capital or settings.initial_capital

        self._commission = commission_model or CommissionModel(
            commission_type=CommissionType.PERCENTAGE,
            percentage_rate=0.001,
        )
        self._slippage = slippage_model or SlippageModel()

        self._trade_sim = TradeSimulator(self._commission, self._slippage)
        self._portfolio = PortfolioSimulator(self._initial_capital, settings.symbol)

        self._trade_log: list[dict[str, Any]] = []
        self._regime_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        *,
        benchmark_series: pd.Series | None = None,
        strategy_manager: Any | None = None,
        risk_manager: Any | None = None,
        regime_detector: Any | None = None,
    ) -> dict[str, Any]:
        """Execute a full backtest on historical data.

        Args:
            df: OHLCV DataFrame with columns: timestamp, open, high, low,
                close, volume.
            benchmark_series: Optional benchmark price series for comparison.
            strategy_manager: Optional StrategyManager instance.
            risk_manager: Optional RiskManager instance.
            regime_detector: Optional RegimeDetector instance.

        Returns:
            Comprehensive backtest results dictionary.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("  BACKTEST ENGINE — Starting")
        logger.info("  Symbol: {} | Bars: {} | Capital: ${:,.2f}",
                     self._settings.symbol, len(df), self._initial_capital)
        logger.info("=" * 60)

        df_feat = self._prepare_features(df)
        regimes = self._detect_regimes(df_feat, regime_detector)
        self._run_simulation(df_feat, regimes, strategy_manager, risk_manager)

        elapsed = time.time() - start_time
        result = self._build_results(benchmark_series, elapsed)

        logger.info("=" * 60)
        logger.info("  BACKTEST COMPLETE — {:.2f}s", elapsed)
        logger.info("  Total Return: {:.2f}% | Sharpe: {:.2f} | Trades: {}",
                     result.get("total_return", 0),
                     result.get("sharpe_ratio", 0),
                     result.get("total_trades", 0))
        logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering on the raw OHLCV data."""
        logger.info("Computing features...")
        df_feat = compute_all_features(df.copy())
        df_feat = df_feat.dropna().reset_index(drop=True)
        logger.info("Feature engineering complete: {} rows, {} columns",
                     len(df_feat), len(df_feat.columns))
        return df_feat

    def _detect_regimes(
        self,
        df: pd.DataFrame,
        detector: Any | None,
    ) -> list[str]:
        """Detect market regimes for each bar.

        If a detector is provided it is used; otherwise a simple
        moving-average-based heuristic classifies regimes.
        """
        logger.info("Detecting market regimes...")

        if detector is not None:
            try:
                results = detector.detect_batch(df)
                regimes = [r.get("regime", "Sideways Market") for r in results]
                self._regime_log = results
                logger.info("Regime detection via HMM complete")
                return regimes
            except Exception as exc:
                logger.warning("HMM regime detection failed ({}), using heuristic", exc)

        regimes = self._heuristic_regimes(df)
        logger.info("Heuristic regime detection complete")
        return regimes

    def _heuristic_regimes(self, df: pd.DataFrame) -> list[str]:
        """Simple heuristic regime detection using moving averages."""
        regimes: list[str] = []

        ema_short = df["close"].ewm(span=20, adjust=False).mean()
        ema_long = df["close"].ewm(span=50, adjust=False).mean()

        for i in range(len(df)):
            if i < 50:
                regimes.append("Sideways Market")
                continue

            close = df["close"].iloc[i]
            es = ema_short.iloc[i]
            el = ema_long.iloc[i]

            vol = df.get("atr", pd.Series(dtype=float))
            if len(vol) > i and vol.iloc[i] > 0:
                atr_pct = vol.iloc[i] / close * 100 if close > 0 else 0
            else:
                atr_pct = 0

            if atr_pct > 3.0:
                regimes.append("High Volatility")
            elif es > el and close > es:
                regimes.append("Bull Market")
            elif es < el and close < es:
                regimes.append("Bear Market")
            else:
                regimes.append("Sideways Market")

        return regimes

    def _run_simulation(
        self,
        df: pd.DataFrame,
        regimes: list[str],
        strategy_manager: Any | None,
        risk_manager: Any | None,
    ) -> None:
        """Bar-by-bar simulation loop."""
        logger.info("Running bar-by-bar simulation...")
        symbol = self._settings.symbol
        total_bars = len(df)

        for i in range(total_bars):
            row = df.iloc[i]
            ts = row.get("timestamp", datetime.now(timezone.utc))
            if isinstance(ts, str):
                ts = pd.Timestamp(ts)
            if not isinstance(ts, datetime):
                ts = ts.to_pydatetime()

            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])
            bar_volume = float(row.get("volume", 0))
            regime = regimes[i] if i < len(regimes) else "Sideways Market"

            closed = self._portfolio.check_stops(
                symbol, bar_high, bar_low, bar_close, ts,
            )
            if closed:
                self._trade_log.append(closed.to_dict())

            if i < 50:
                self._portfolio.snapshot(ts, bar_close)
                continue

            signal = self._generate_signal(
                row, regime, strategy_manager, risk_manager,
            )

            if signal and signal.get("action") in ("BUY", "SHORT"):
                self._execute_signal(signal, row, ts, bar_volume)

            elif signal and signal.get("action") in ("SELL", "COVER"):
                self._close_signal(signal, row, ts, bar_volume)

            self._portfolio.snapshot(ts, bar_close)

            if (i + 1) % 500 == 0:
                logger.info("Processed {}/{} bars", i + 1, total_bars)

        self._close_all_positions(df, regimes)

    def _generate_signal(
        self,
        row: pd.Series,
        regime: str,
        strategy_manager: Any | None,
        risk_manager: Any | None,
    ) -> dict[str, Any] | None:
        """Generate a trade signal for the current bar."""
        if self._portfolio.positions:
            return None

        if strategy_manager is not None:
            try:
                from portfolio.portfolio_state import PortfolioState
                from portfolio.account import Account

                result = strategy_manager.generate_trade_signal(
                    row, regime, confidence=0.8,
                    risk_manager=risk_manager,
                )

                if isinstance(result, dict):
                    sig_type = result.get("signal", "HOLD")
                    if sig_type in ("BUY", "SHORT"):
                        return {
                            "action": sig_type,
                            "entry": result.get("entry", float(row["close"])),
                            "stop_loss": result.get("stop_loss", 0.0),
                            "take_profit": result.get("take_profit", 0.0),
                            "position_size": result.get("position_size", 0.0),
                        }
                    elif sig_type in ("SELL", "COVER"):
                        return {"action": sig_type}
            except Exception as exc:
                logger.debug("Strategy manager error: {}", exc)

        return self._heuristic_signal(row, regime)

    def _heuristic_signal(
        self,
        row: pd.Series,
        regime: str,
    ) -> dict[str, Any] | None:
        """Simple momentum-based signal for standalone backtesting."""
        close = float(row["close"])
        atr = float(row.get("atr", close * 0.02))

        rsi = row.get("rsi", 50.0)
        if pd.isna(rsi):
            rsi = 50.0

        macd = row.get("macd", 0.0)
        macd_signal = row.get("macd_signal", 0.0)
        if pd.isna(macd):
            macd = 0.0
        if pd.isna(macd_signal):
            macd_signal = 0.0

        if "bull" in regime.lower():
            if rsi < 70 and macd > macd_signal:
                risk_pct = self._settings.strategy.risk_percentage / 100.0
                risk_amount = self._portfolio.cash * risk_pct
                stop_distance = atr * self._settings.strategy.atr_multiplier_stop_loss
                if stop_distance > 0:
                    qty = risk_amount / stop_distance
                else:
                    qty = 0

                if qty > 0:
                    return {
                        "action": "BUY",
                        "entry": close,
                        "stop_loss": close - stop_distance,
                        "take_profit": close + atr * self._settings.strategy.atr_multiplier_take_profit,
                        "position_size": qty,
                    }

        elif "bear" in regime.lower():
            if rsi > 30 and macd < macd_signal:
                risk_pct = self._settings.strategy.risk_percentage / 100.0
                risk_amount = self._portfolio.cash * risk_pct
                stop_distance = atr * self._settings.strategy.atr_multiplier_stop_loss
                if stop_distance > 0:
                    qty = risk_amount / stop_distance
                else:
                    qty = 0

                if qty > 0:
                    return {
                        "action": "SHORT",
                        "entry": close,
                        "stop_loss": close + stop_distance,
                        "take_profit": close - atr * self._settings.strategy.atr_multiplier_take_profit,
                        "position_size": qty,
                    }

        return None

    def _execute_signal(
        self,
        signal: dict[str, Any],
        row: pd.Series,
        timestamp: datetime,
        bar_volume: float,
    ) -> None:
        """Execute an entry signal."""
        action = signal["action"]
        qty = signal.get("position_size", 0)
        if qty <= 0:
            return

        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        order = SimulatedOrder(
            symbol=self._settings.symbol,
            side=side,
            order_type=SimOrderType.MARKET,
            quantity=qty,
            timestamp=timestamp,
        )

        fill = self._trade_sim.execute(
            order,
            bar_open=float(row["open"]),
            bar_high=float(row["high"]),
            bar_low=float(row["low"]),
            bar_close=float(row["close"]),
            bar_volume=bar_volume,
            timestamp=timestamp,
        )

        if fill.status in (FillStatus.FILLED, FillStatus.PARTIAL):
            self._portfolio.process_fill(
                fill,
                stop_loss=signal.get("stop_loss", 0.0),
                take_profit=signal.get("take_profit", 0.0),
            )

    def _close_signal(
        self,
        signal: dict[str, Any],
        row: pd.Series,
        timestamp: datetime,
        bar_volume: float,
    ) -> None:
        """Execute an exit signal."""
        symbol = self._settings.symbol
        pos = self._portfolio.positions.get(symbol)
        if not pos:
            return

        side = OrderSide.SELL if pos.direction == "LONG" else OrderSide.BUY
        order = SimulatedOrder(
            symbol=symbol,
            side=side,
            order_type=SimOrderType.MARKET,
            quantity=pos.quantity,
            timestamp=timestamp,
        )

        fill = self._trade_sim.execute(
            order,
            bar_open=float(row["open"]),
            bar_high=float(row["high"]),
            bar_low=float(row["low"]),
            bar_close=float(row["close"]),
            bar_volume=bar_volume,
            timestamp=timestamp,
        )

        if fill.status in (FillStatus.FILLED, FillStatus.PARTIAL):
            self._portfolio.process_fill(fill)

    def _close_all_positions(self, df: pd.DataFrame, regimes: list[str]) -> None:
        """Close all remaining open positions at the last bar."""
        if not self._portfolio.positions:
            return

        last_row = df.iloc[-1]
        ts = last_row.get("timestamp", datetime.now(timezone.utc))
        if not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()

        for symbol, pos in list(self._portfolio.positions.items()):
            closed = self._portfolio._force_close(
                pos, float(last_row["close"]), ts, "backtest_end",
            )
            self._trade_log.append(closed.to_dict())

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _build_results(
        self,
        benchmark_series: pd.Series | None,
        elapsed: float,
    ) -> dict[str, Any]:
        """Build the final results dictionary."""
        equity = self._portfolio.get_equity_series()
        trades = [t.to_dict() for t in self._portfolio.closed_trades]

        bench = None
        if benchmark_series is not None and len(benchmark_series) > 1:
            ratio = self._initial_capital / benchmark_series.iloc[0]
            bench = benchmark_series * ratio

        analyzer = PerformanceAnalyzer(
            equity_series=equity,
            trades=trades,
            benchmark_series=bench,
            initial_capital=self._initial_capital,
        )

        summary = analyzer.summary()
        full = analyzer.compute_all()

        result = {
            **summary,
            "analytics": full,
            "equity_series": equity,
            "trades": trades,
            "regime_log": self._regime_log,
            "portfolio_summary": self._portfolio.get_summary(),
            "execution_time_seconds": round(elapsed, 2),
        }

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> PortfolioSimulator:
        """The underlying portfolio simulator."""
        return self._portfolio


def run_backtest(
    settings: Settings | None = None,
    df: pd.DataFrame | None = None,
    *,
    initial_capital: float = 100_000.0,
    benchmark_series: pd.Series | None = None,
    commission_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    strategy_manager: Any | None = None,
    risk_manager: Any | None = None,
    regime_detector: Any | None = None,
) -> dict[str, Any]:
    """Run a complete backtest — top-level convenience function.

    Args:
        settings: Application settings. If None, default settings are used.
        df: OHLCV DataFrame. If None, data is loaded via DataManager.
        initial_capital: Starting portfolio value.
        benchmark_series: Optional benchmark for comparison.
        commission_rate: Commission as fraction of notional.
        slippage_pct: Slippage as fraction of price.
        strategy_manager: Optional StrategyManager instance.
        risk_manager: Optional RiskManager instance.
        regime_detector: Optional RegimeDetector instance.

    Returns:
        Comprehensive backtest results dictionary.
    """
    if settings is None:
        from config.settings import Settings
        settings = Settings()
        settings.initial_capital = initial_capital

    if df is None:
        from data_loader.data_manager import DataManager
        manager = DataManager(settings)
        df, _ = manager.get_data()

    commission = CommissionModel(
        commission_type=CommissionType.PERCENTAGE,
        percentage_rate=commission_rate,
    )
    slippage = SlippageModel(fixed_pct=slippage_pct)

    engine = BacktestEngine(
        settings=settings,
        commission_model=commission,
        slippage_model=slippage,
        initial_capital=initial_capital,
    )

    return engine.run(
        df,
        benchmark_series=benchmark_series,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        regime_detector=regime_detector,
    )
