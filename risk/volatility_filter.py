"""Volatility Filter.

Rejects trades during extreme volatility conditions like huge ATR spikes
or unexpected price gaps.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from signals.signal import TradeSignal


class VolatilityFilter:
    """Evaluates market volatility conditions for safe trading."""

    def is_safe(
        self,
        signal: TradeSignal,
        latest_data: pd.Series,
        historical_atr_mean: float | None = None,
    ) -> bool:
        """Check if current volatility allows for safe execution.

        Args:
            signal: The proposed trade signal.
            latest_data: The most recent bar with engineered features.
            historical_atr_mean: The mean ATR over a long lookback.

        Returns:
            True if volatility is safe, False if extreme.
        """
        current_atr = latest_data.get("atr", 0.0)
        
        # 1. Extreme ATR Spike Check
        # Reject if current ATR is > 3x the historical mean
        if historical_atr_mean and historical_atr_mean > 0:
            if current_atr > historical_atr_mean * 3.0:
                logger.warning(
                    "Volatility Filter Reject: ATR Spike ({:.2f} > 3x historical mean {:.2f})",
                    current_atr,
                    historical_atr_mean,
                )
                return False

        # 2. Gap Check
        # Reject if the candle opened with a massive gap (> 2% gap from previous close).
        # We assume the caller provides 'prev_close' in latest_data if they want this check.
        if "prev_close" in latest_data:
            gap_pct = abs(latest_data["open"] - latest_data["prev_close"]) / latest_data["prev_close"]
            if gap_pct > 0.02:
                logger.warning(
                    "Volatility Filter Reject: Extreme Price Gap ({:.2%})",
                    gap_pct,
                )
                return False

        return True
