"""Trade filtering and rejection logic.

Validates signals against strict risk management rules: session hours,
duplicate signals, low confidence, and excessive volatility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from config.settings import StrategySettings
from signals.signal import TradeSignal


class TradeFilter:
    """Evaluate whether a generated signal should be allowed to execute.

    Args:
        settings: Application strategy settings.
    """

    def __init__(self, settings: StrategySettings) -> None:
        self.settings = settings
        self._daily_trade_count: dict[str, int] = {}
        self._last_signal_direction: str | None = None

    def is_valid(
        self,
        signal: TradeSignal,
        current_spread_pct: float = 0.0,
        max_spread_pct: float = 0.005,
    ) -> bool:
        """Run all filter checks on a trade signal.

        Args:
            signal: The candidate trade signal.
            current_spread_pct: Current asset spread as a percentage.
            max_spread_pct: Maximum allowed spread percentage.

        Returns:
            True if the signal passes all checks, False otherwise.
        """
        # 1. Confidence check
        if signal.confidence < self.settings.min_regime_confidence:
            logger.info(
                "Filter Reject: Low confidence ({:.2f} < {:.2f})",
                signal.confidence,
                self.settings.min_regime_confidence,
            )
            return False

        # 2. Session check (UTC hour)
        hour = signal.timestamp.hour
        start = self.settings.session_start_hour
        end = self.settings.session_end_hour
        if start <= end:
            in_session = start <= hour <= end
        else:
            in_session = hour >= start or hour <= end

        if not in_session:
            logger.info("Filter Reject: Outside trading session (Hour: {})", hour)
            return False

        # 3. Spread check
        if current_spread_pct > max_spread_pct:
            logger.info(
                "Filter Reject: Spread too high ({:.2%} > {:.2%})",
                current_spread_pct,
                max_spread_pct,
            )
            return False

        # 4. Duplicate signal check
        if signal.signal.value == self._last_signal_direction:
            logger.info(
                "Filter Reject: Duplicate signal direction ({})",
                signal.signal.value,
            )
            return False

        # 5. Max trades per day
        date_str = signal.timestamp.strftime("%Y-%m-%d")
        trades_today = self._daily_trade_count.get(date_str, 0)
        if trades_today >= self.settings.max_trades_per_day:
            logger.info(
                "Filter Reject: Max trades reached for today ({})",
                trades_today,
            )
            return False

        return True

    def record_execution(self, signal: TradeSignal) -> None:
        """Mark a signal as executed to update internal counters.

        Args:
            signal: The executed signal.
        """
        date_str = signal.timestamp.strftime("%Y-%m-%d")
        self._daily_trade_count[date_str] = self._daily_trade_count.get(date_str, 0) + 1
        self._last_signal_direction = signal.signal.value
