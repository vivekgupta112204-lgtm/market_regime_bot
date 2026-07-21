"""Stop Loss Engine.

Calculates initial and dynamic stop losses.
Supports Fixed, ATR, Swing High/Low, Percentage, Break-even, and Trailing stops.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from signals.signal import SignalType


class StopLossEngine:
    """Computes stop loss levels using various methodologies."""

    def calculate_initial_stop(
        self,
        latest_data: pd.Series,
        entry_price: float,
        action: SignalType,
        method: str = "atr",
        atr_multiplier: float = 2.0,
        percentage: float = 0.02,
    ) -> float:
        """Calculate the initial stop loss price.

        Args:
            latest_data: Most recent OHLCV bar.
            entry_price: Planned entry price.
            action: BUY or SHORT.
            method: 'atr', 'percentage', 'swing'.
            atr_multiplier: Multiplier for ATR stops.
            percentage: Percentage drop for percentage stops.

        Returns:
            The calculated stop loss price.
        """
        if method == "percentage":
            dist = entry_price * percentage
            return entry_price - dist if action == SignalType.BUY else entry_price + dist

        elif method == "swing":
            # Assumes candle anatomy features are available
            if action == SignalType.BUY:
                return latest_data["low"] - (latest_data.get("atr", 0) * 0.1)
            else:
                return latest_data["high"] + (latest_data.get("atr", 0) * 0.1)

        else:
            # Default: ATR
            atr = latest_data.get("atr", 0.0)
            dist = atr * atr_multiplier
            return entry_price - dist if action == SignalType.BUY else entry_price + dist

    def calculate_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        entry_price: float,
        action: SignalType,
        atr: float,
        trailing_activation_rr: float = 1.0,
        trailing_distance_atr: float = 1.5,
        initial_risk: float = 0.0,
    ) -> float:
        """Calculate a dynamic trailing stop.

        Args:
            current_price: The current market price.
            current_stop: The active stop loss.
            entry_price: The original entry price.
            action: BUY or SHORT direction.
            atr: Current ATR.
            trailing_activation_rr: RR threshold to activate trail.
            trailing_distance_atr: Distance to trail in ATR.
            initial_risk: The original fiat risk per unit.

        Returns:
            The updated stop loss (moves only in favorable direction).
        """
        if initial_risk <= 0:
            return current_stop

        # Check if we should activate trailing
        if action == SignalType.BUY:
            current_reward = current_price - entry_price
            if current_reward / initial_risk >= trailing_activation_rr:
                new_stop = current_price - (atr * trailing_distance_atr)
                return max(current_stop, new_stop) # Only move up
        else:
            current_reward = entry_price - current_price
            if current_reward / initial_risk >= trailing_activation_rr:
                new_stop = current_price + (atr * trailing_distance_atr)
                return min(current_stop, new_stop) # Only move down
                
        return current_stop
