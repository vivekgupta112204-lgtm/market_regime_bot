"""Sideways / Choppy Market strategy.

Uses RSI, Bollinger Bands, and CCI for mean-reversion trading.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import StrategySettings
from portfolio.position import PositionSizer
from signals.signal import SignalType
from strategy.base_strategy import BaseStrategy


class SidewaysStrategy(BaseStrategy):
    """Mean Reversion strategy for Sideways regimes."""

    def __init__(self, settings: StrategySettings) -> None:
        super().__init__(settings, name="MeanReversion")
        self.sizer = PositionSizer(settings)

    def initialize(self) -> None:
        logger.debug(f"{self.name} initialized.")

    def _evaluate_conditions(self, data: pd.Series) -> tuple[SignalType, list[str]]:
        if self.should_enter(data):
            reasons = [
                "Price near/below Lower Bollinger Band",
                "RSI < 30 (Oversold)",
                "CCI < -100",
            ]
            return SignalType.BUY, reasons

        if self.should_exit(data):
            reasons = [
                "Price near/above Upper Bollinger Band",
                "RSI > 70 (Overbought)",
            ]
            return SignalType.SELL, reasons

        return SignalType.HOLD, ["Awaiting mean reversion setup"]

    def should_enter(self, data: pd.Series) -> bool:
        """Entry rules: Close < BB Lower, RSI < 30, CCI < -100."""
        try:
            close = data["close"]
            bb_lower = data["bb_lower"]
            rsi = data["rsi"]
            cci = data["cci"]

            return bool(close <= bb_lower * 1.005 and rsi < 30 and cci < -100)
        except KeyError as e:
            logger.error(f"Missing feature for {self.name}: {e}")
            return False

    def should_exit(self, data: pd.Series) -> bool:
        """Exit rules: Close > BB Upper, RSI > 70."""
        try:
            close = data["close"]
            bb_upper = data["bb_upper"]
            rsi = data["rsi"]

            return bool(close >= bb_upper * 0.995 or rsi > 70)
        except KeyError:
            return False

    def calculate_position_size(
        self, capital: float, entry: float, stop: float
    ) -> float:
        """Use Kelly Criterion if available, else fallback to fixed fractional."""
        return self.sizer.calculate_size(
            capital=capital,
            entry_price=entry,
            stop_loss=stop,
            method="fixed_fractional",  # Defaulting to fixed since we lack history initially
        )
