"""Strategy orchestration and signal generation engine.

Receives regime state and market data, activates the appropriate
sub-strategy, applies trade filters, and produces a final TradeSignal.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings
from portfolio.exposure import TradeFilter
from portfolio.portfolio_state import PortfolioState
from signals.signal import SignalType, TradeSignal
from risk.risk_manager import RiskManager
from strategy.base_strategy import BaseStrategy
from strategy.bear_strategy import BearStrategy
from strategy.breakout_strategy import BreakoutStrategy
from strategy.bull_strategy import BullStrategy
from strategy.sideways_strategy import SidewaysStrategy
from strategy.volatility_strategy import VolatilityStrategy


class StrategyManager:
    """Manages strategy routing and signal validation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
        # Instantiate all strategies.
        self.strategies: dict[str, BaseStrategy] = {
            "bull": BullStrategy(settings.strategy),
            "bear": BearStrategy(settings.strategy),
            "sideways": SidewaysStrategy(settings.strategy),
            "high_volatility": VolatilityStrategy(settings.strategy),
            "low_volatility": BreakoutStrategy(settings.strategy),
        }
        
        for strat in self.strategies.values():
            strat.initialize()
            
        self.trade_filter = TradeFilter(settings.strategy)

    def _map_regime_to_strategy_key(self, regime: str) -> str:
        """Map the string regime name to a strategy dictionary key."""
        r = regime.lower()
        if "bull" in r:
            return "bull"
        if "bear" in r:
            return "bear"
        if "sideways" in r:
            return "sideways"
        if "high" in r and "volatility" in r:
            return "high_volatility"
        if "low" in r and "volatility" in r:
            return "low_volatility"
        
        logger.warning(f"Unknown regime '{regime}', defaulting to sideways")
        return "sideways"

    def generate_trade_signal(
        self,
        latest_data: pd.DataFrame | pd.Series,
        regime: str,
        confidence: float,
        current_spread_pct: float = 0.0,
        risk_manager: RiskManager | None = None,
        portfolio_state: PortfolioState | None = None,
    ) -> dict[str, Any]:
        """Generate a trading signal for the current market state.

        Args:
            latest_data: The latest OHLCV bar with all engineered features.
            regime: Current detected regime (e.g., "Bull Market").
            confidence: Regime detection confidence (0.0 to 1.0).
            current_spread_pct: Current spread for filter checks.
            risk_manager: Optional Phase 5 RiskManager.
            portfolio_state: Optional Phase 5 PortfolioState.

        Returns:
            JSON-serializable dictionary of the TradeSignal, or RiskManager output.
        """
        # Ensure we're working with a Series.
        if isinstance(latest_data, pd.DataFrame):
            row = latest_data.iloc[-1]
        else:
            row = latest_data
            
        # Confidence Floor Gate
        import datetime # Import in case needed
        if confidence < 0.60:
            logger.warning(f"Regime {regime} predicted but dominant confidence ({confidence:.2f}) is below 0.60 floor. Forcing HOLD.")
            ts = row.name if hasattr(row, 'name') and row.name else pd.Timestamp.utcnow()
            return TradeSignal.create_hold(
                timestamp=ts,
                regime=regime,
                strategy="ALL",
                reason=f"Confidence {confidence:.2f} < 0.60 floor threshold.",
            ).to_dict()
            
        # 1. Route to appropriate strategy
        strat_key = self._map_regime_to_strategy_key(regime)
        active_strategy = self.strategies[strat_key]
        
        # 2. Generate raw signal
        signal = active_strategy.generate_signal(row, regime, confidence)
        
        # 3. Phase 5 Integration
        if risk_manager and portfolio_state:
            # Delegate to Phase 5 Risk Management Engine
            return risk_manager.evaluate_trade(signal, row, portfolio_state)
            
        # 4. Phase 4 Legacy Logic (If RiskManager not provided)
        if signal.signal != SignalType.HOLD:
            is_valid = self.trade_filter.is_valid(signal, current_spread_pct)
            if not is_valid:
                signal = TradeSignal.create_hold(
                    timestamp=signal.timestamp,
                    regime=signal.regime,
                    strategy=active_strategy.name,
                    reason=f"Rejected by TradeFilter. Original: {signal.signal.value}",
                )
            else:
                self.trade_filter.record_execution(signal)
                
        if signal.signal in (SignalType.BUY, SignalType.SHORT):
            capital = self.settings.initial_capital
            size = active_strategy.calculate_position_size(
                capital=capital,
                entry=signal.entry,
                stop=signal.stop_loss,
            )
            signal.reason.append(f"Calculated Size: {size}")
            
            if size <= 0:
                signal = TradeSignal.create_hold(
                    timestamp=signal.timestamp,
                    regime=signal.regime,
                    strategy=active_strategy.name,
                    reason="Position sizer returned 0 (Risk limit or min lot size).",
                )

        return signal.to_dict()
