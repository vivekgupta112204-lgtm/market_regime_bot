"""Take Profit Engine.

Calculates static and dynamic take profit levels.
"""

from __future__ import annotations

import pandas as pd
from signals.signal import SignalType


class TakeProfitEngine:
    """Computes take profit target levels."""

    def calculate_target(
        self,
        latest_data: pd.Series,
        entry_price: float,
        stop_loss: float,
        action: SignalType,
        method: str = "fixed_rr",
        rr_ratio: float = 2.0,
        atr_multiplier: float = 4.0,
    ) -> float:
        """Calculate the take profit price.

        Args:
            latest_data: Most recent OHLCV bar.
            entry_price: Planned entry price.
            stop_loss: Calculated stop loss.
            action: BUY or SHORT.
            method: 'fixed_rr', 'atr'.
            rr_ratio: Desired Risk/Reward ratio.
            atr_multiplier: Multiplier for ATR targets.

        Returns:
            The calculated take profit price.
        """
        if method == "fixed_rr":
            risk = abs(entry_price - stop_loss)
            reward = risk * rr_ratio
            return entry_price + reward if action == SignalType.BUY else entry_price - reward

        else:
            # Default: ATR
            atr = latest_data.get("atr", 0.0)
            dist = atr * atr_multiplier
            return entry_price + dist if action == SignalType.BUY else entry_price - dist
