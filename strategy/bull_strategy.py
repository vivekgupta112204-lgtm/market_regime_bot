"""Bull Market trend-following strategy.

Uses EMA crosses, MACD, and ADX to ride strong upward trends.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import StrategySettings
from portfolio.position import PositionSizer
from signals.signal import SignalType
from strategy.base_strategy import BaseStrategy


class BullStrategy(BaseStrategy):
    """Trend-following strategy for Bull regimes."""

    def __init__(self, settings: StrategySettings) -> None:
        super().__init__(settings, name="BullTrendFollowing")
        self.sizer = PositionSizer(settings)

    def initialize(self) -> None:
        logger.debug(f"{self.name} initialized.")

    def _evaluate_conditions(self, data: pd.Series) -> tuple[SignalType, list[str]]:
        if self.should_enter(data):
            reasons = [
                "EMA20 > EMA50",
                "MACD Bullish (Hist > 0)",
                "ADX > 25 (Strong Trend)",
            ]
            return SignalType.BUY, reasons

        if self.should_exit(data):
            reasons = ["EMA20 < EMA50 (Trend Reversal)"]
            return SignalType.SELL, reasons

        return SignalType.HOLD, ["Trend unchanged / waiting for setup"]

    def should_enter(self, data: pd.Series) -> bool:
        """Entry rules: EMA20 > EMA50, MACD Hist > 0, ADX > 25."""
        try:
            ema20 = data["ema_20"]
            ema50 = data["ema_50"]
            macd_hist = data["macd_histogram"]
            adx = data["adx"]

            return bool(ema20 > ema50 and macd_hist > 0 and adx > 25)
        except KeyError as e:
            logger.error(f"Missing feature for {self.name}: {e}")
            return False

    def should_exit(self, data: pd.Series) -> bool:
        """Exit rules: EMA20 < EMA50."""
        try:
            return bool(data["ema_20"] < data["ema_50"])
        except KeyError:
            return False

    def calculate_position_size(
        self, capital: float, entry: float, stop: float
    ) -> float:
        """Use ATR risk sizing for trend-following."""
        return self.sizer.calculate_size(
            capital=capital,
            entry_price=entry,
            stop_loss=stop,
            method="atr_risk",
        )
