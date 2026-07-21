"""Abstract base class for all trading strategies.

Enforces a strict interface for signal generation, condition
checking, and risk management calculations.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import StrategySettings
from signals.signal import SignalType, TradeSignal


class BaseStrategy(abc.ABC):
    """Abstract trading strategy interface."""

    def __init__(self, settings: StrategySettings, name: str) -> None:
        self.settings = settings
        self.name = name

    @abc.abstractmethod
    def initialize(self) -> None:
        """Called once before the strategy begins evaluating."""
        pass

    def generate_signal(
        self,
        latest_data: pd.Series,
        regime: str,
        confidence: float,
    ) -> TradeSignal:
        """Core signal generation template method.

        Args:
            latest_data: The most recent bar with engineered features.
            regime: Current detected market regime.
            confidence: Regime detection confidence.

        Returns:
            A populated ``TradeSignal``.
        """
        timestamp: datetime = latest_data["timestamp"]
        close_price: float = latest_data["close"]

        # Default reason/action
        action = SignalType.HOLD
        reason = []

        if self.should_enter(latest_data):
            action = SignalType.BUY if "bull" in regime.lower() else SignalType.SHORT
            # Some strategies might dictate specific direction regardless of regime name
            # Let the specific strategy override `should_enter` appropriately.
            
        elif self.should_exit(latest_data):
            action = SignalType.SELL if "bull" in regime.lower() else SignalType.COVER

        # Recalculate explicitly to let subclasses define exact long/short logic
        action, reason = self._evaluate_conditions(latest_data)

        if action in (SignalType.BUY, SignalType.SHORT):
            stop_loss = self.calculate_stop_loss(latest_data, action, close_price)
            take_profit = self.calculate_take_profit(latest_data, action, close_price)
            risk = abs(close_price - stop_loss)
            reward = abs(take_profit - close_price)
            risk_reward = reward / risk if risk > 0 else 0.0

            return TradeSignal(
                timestamp=timestamp,
                regime=regime,
                strategy=self.name,
                signal=action,
                confidence=confidence,
                entry=close_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=round(risk_reward, 2),
                reason=reason,
            )

        if action in (SignalType.SELL, SignalType.COVER):
            return TradeSignal(
                timestamp=timestamp,
                regime=regime,
                strategy=self.name,
                signal=action,
                confidence=confidence,
                entry=close_price,
                reason=reason,
            )

        return TradeSignal.create_hold(timestamp, regime, self.name, "No edge detected")

    @abc.abstractmethod
    def _evaluate_conditions(self, data: pd.Series) -> tuple[SignalType, list[str]]:
        """Evaluate technical conditions and return action and reason."""
        pass

    @abc.abstractmethod
    def should_enter(self, data: pd.Series) -> bool:
        """Return True if entry conditions are met."""
        pass

    @abc.abstractmethod
    def should_exit(self, data: pd.Series) -> bool:
        """Return True if exit/close conditions are met."""
        pass

    def calculate_stop_loss(
        self, data: pd.Series, action: SignalType, entry_price: float
    ) -> float:
        """Calculate stop loss based on ATR.

        Args:
            data: Latest market data with ATR feature.
            action: BUY or SHORT.
            entry_price: The planned entry price.

        Returns:
            The calculated stop loss price.
        """
        atr = data.get("atr", 0.0)
        distance = atr * self.settings.atr_multiplier_stop_loss

        if action == SignalType.BUY:
            return entry_price - distance
        elif action == SignalType.SHORT:
            return entry_price + distance
        return entry_price

    def calculate_take_profit(
        self, data: pd.Series, action: SignalType, entry_price: float
    ) -> float:
        """Calculate take profit based on ATR.

        Args:
            data: Latest market data with ATR feature.
            action: BUY or SHORT.
            entry_price: The planned entry price.

        Returns:
            The calculated take profit price.
        """
        atr = data.get("atr", 0.0)
        distance = atr * self.settings.atr_multiplier_take_profit

        if action == SignalType.BUY:
            return entry_price + distance
        elif action == SignalType.SHORT:
            return entry_price - distance
        return entry_price

    @abc.abstractmethod
    def calculate_position_size(self, capital: float, entry: float, stop: float) -> float:
        """Calculate position size for this specific strategy.
        
        Subclasses can delegate this to PositionSizer or implement custom logic.
        """
        pass
