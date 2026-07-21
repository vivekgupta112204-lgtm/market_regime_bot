"""Low Volatility breakout strategy.

Uses Donchian Channels, Volume, and ATR to catch explosive moves out
of tight consolidation.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import StrategySettings
from portfolio.position import PositionSizer
from signals.signal import SignalType
from strategy.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy for Low Volatility regimes."""

    def __init__(self, settings: StrategySettings) -> None:
        super().__init__(settings, name="LowVolBreakout")
        self.sizer = PositionSizer(settings)

    def initialize(self) -> None:
        logger.debug(f"{self.name} initialized.")

    def _evaluate_conditions(self, data: pd.Series) -> tuple[SignalType, list[str]]:
        # Donchian channel breakout
        if self.should_enter(data):
            if data["close"] >= data["donchian_upper"]:
                return SignalType.BUY, ["Close above Donchian Upper", "Volume Surge"]
            else:
                return SignalType.SHORT, ["Close below Donchian Lower", "Volume Surge"]

        if self.should_exit(data):
            return SignalType.SELL, ["Breakout failed / Reverted to EMA20"]

        return SignalType.HOLD, ["Consolidating within Donchian Channel"]

    def should_enter(self, data: pd.Series) -> bool:
        """Entry rules: Price > Donchian Upper OR < Donchian Lower, with Volume surge."""
        try:
            close = data["close"]
            d_upper = data["donchian_upper"]
            d_lower = data["donchian_lower"]
            vol_change = data["volume_change"]

            breakout = (close >= d_upper * 0.999) or (close <= d_lower * 1.001)
            vol_surge = vol_change > 0.5  # 50% volume spike

            return bool(breakout and vol_surge)
        except KeyError as e:
            logger.error(f"Missing feature for {self.name}: {e}")
            return False

    def should_exit(self, data: pd.Series) -> bool:
        """Exit rules: Price falls back past short EMA."""
        try:
            return bool(
                abs(data["close"] - data["ema_20"]) / data["close"] < 0.005
            )
        except KeyError:
            return False

    def calculate_position_size(
        self, capital: float, entry: float, stop: float
    ) -> float:
        """Use ATR risk sizing."""
        return self.sizer.calculate_size(
            capital=capital,
            entry_price=entry,
            stop_loss=stop,
            method="atr_risk",
        )
