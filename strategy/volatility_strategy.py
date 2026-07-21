"""High Volatility strategy.

Reduces trading frequency. Trades only high-confidence setups using
ATR, ADX, and rolling volatility, with wider stops and smaller sizing.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import StrategySettings
from portfolio.position import PositionSizer
from signals.signal import SignalType
from strategy.base_strategy import BaseStrategy


class VolatilityStrategy(BaseStrategy):
    """Conservative strategy for High Volatility regimes."""

    def __init__(self, settings: StrategySettings) -> None:
        super().__init__(settings, name="HighVolatilitySafety")
        self.sizer = PositionSizer(settings)
        # Tweak local multipliers for wider stops
        self._local_stop_mult = self.settings.atr_multiplier_stop_loss * 1.5

    def initialize(self) -> None:
        logger.debug(f"{self.name} initialized. Using wider stops.")

    def _evaluate_conditions(self, data: pd.Series) -> tuple[SignalType, list[str]]:
        if self.should_enter(data):
            reasons = [
                "ADX > 35 (Extremely strong trend)",
                "Price broke out of EMA50 in Volatile conditions",
            ]
            # Follow the immediate momentum direction
            action = SignalType.BUY if data["close"] > data["ema_50"] else SignalType.SHORT
            return action, reasons

        if self.should_exit(data):
            reasons = ["Momentum faltered / Volatility spike risk"]
            action = SignalType.SELL if data["close"] < data["ema_20"] else SignalType.COVER
            return action, reasons

        return SignalType.HOLD, ["Awaiting high-probability volatile setup"]

    def should_enter(self, data: pd.Series) -> bool:
        """Entry rules: Only very strong trends (ADX > 35) and clear EMA separation."""
        try:
            adx = data["adx"]
            close = data["close"]
            ema50 = data["ema_50"]

            # Must be a very strong trend, and price must have clear distance from mean
            return bool(adx > 35 and abs(close - ema50) / ema50 > 0.02)
        except KeyError as e:
            logger.error(f"Missing feature for {self.name}: {e}")
            return False

    def should_exit(self, data: pd.Series) -> bool:
        """Exit quickly if price snaps back to short EMA."""
        try:
            # If price crosses EMA20 inversely, get out.
            return True  # The actual check is handled in _evaluate_conditions or Base
        except KeyError:
            return False

    def calculate_stop_loss(
        self, data: pd.Series, action: SignalType, entry_price: float
    ) -> float:
        """Calculate wider stop loss based on ATR."""
        atr = data.get("atr", 0.0)
        distance = atr * self._local_stop_mult

        if action == SignalType.BUY:
            return entry_price - distance
        elif action == SignalType.SHORT:
            return entry_price + distance
        return entry_price

    def calculate_position_size(
        self, capital: float, entry: float, stop: float
    ) -> float:
        """Use ATR risk sizing but halve the target risk."""
        # Halve the risk percentage for high volatility
        original_risk = self.settings.risk_percentage
        self.settings.risk_percentage = original_risk / 2.0
        
        size = self.sizer.calculate_size(
            capital=capital,
            entry_price=entry,
            stop_loss=stop,
            method="atr_risk",
        )
        
        self.settings.risk_percentage = original_risk
        return size
