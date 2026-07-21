"""Rolling regime-prediction history.

Maintains a time-ordered log of every regime detection, including the
predicted state, confidence, close price, and return.  The history is
stored in-memory as a list of ``HistoryRecord`` dataclass instances
and can be exported to a pandas DataFrame or persisted to disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class HistoryRecord:
    """A single entry in the regime-prediction history.

    Attributes:
        timestamp: UTC datetime of the prediction.
        state_id: Numeric hidden-state index.
        regime: Human-readable regime label.
        confidence: Dominant-state posterior probability.
        close_price: Closing price of the corresponding bar.
        daily_return: Daily return of the corresponding bar.
        entropy: Shannon entropy of the posterior.
    """

    timestamp: datetime
    state_id: int
    regime: str
    confidence: float
    close_price: float
    daily_return: float
    entropy: float


class RegimeHistory:
    """Rolling in-memory log of regime predictions.

    Args:
        max_records: Maximum number of records to retain.  When
            exceeded the oldest records are evicted.
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._records: list[HistoryRecord] = []
        self._max_records = max_records

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    def add(
        self,
        timestamp: datetime,
        state_id: int,
        regime: str,
        confidence: float,
        close_price: float,
        daily_return: float,
        entropy: float = 0.0,
    ) -> HistoryRecord:
        """Append a new prediction record.

        Args:
            timestamp: Prediction time (UTC).
            state_id: Predicted hidden-state index.
            regime: Regime label.
            confidence: Dominant-state probability.
            close_price: Bar close price.
            daily_return: Bar daily return.
            entropy: Posterior entropy.

        Returns:
            The created ``HistoryRecord``.
        """
        record = HistoryRecord(
            timestamp=timestamp,
            state_id=state_id,
            regime=regime,
            confidence=confidence,
            close_price=close_price,
            daily_return=daily_return,
            entropy=entropy,
        )
        self._records.append(record)

        # Evict oldest if over capacity.
        if len(self._records) > self._max_records:
            evicted = len(self._records) - self._max_records
            self._records = self._records[evicted:]
            logger.debug("History evicted {} oldest records", evicted)

        return record

    def add_bulk(
        self,
        timestamps: list[datetime],
        state_ids: list[int],
        label_map: dict[int, str],
        confidences: list[float],
        close_prices: list[float],
        daily_returns: list[float],
        entropies: list[float] | None = None,
    ) -> int:
        """Append multiple records at once.

        Args:
            timestamps: List of prediction times.
            state_ids: List of predicted state indices.
            label_map: State → label mapping.
            confidences: List of confidence values.
            close_prices: List of close prices.
            daily_returns: List of daily returns.
            entropies: Optional list of entropy values.

        Returns:
            Number of records added.
        """
        n = len(timestamps)
        if entropies is None:
            entropies = [0.0] * n

        for i in range(n):
            self.add(
                timestamp=timestamps[i],
                state_id=int(state_ids[i]),
                regime=label_map.get(int(state_ids[i]), f"State {state_ids[i]}"),
                confidence=confidences[i],
                close_price=close_prices[i],
                daily_return=daily_returns[i],
                entropy=entropies[i],
            )

        logger.info("Added {} records to regime history", n)
        return n

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def latest(self) -> HistoryRecord | None:
        """Return the most recent record, or ``None`` if empty."""
        return self._records[-1] if self._records else None

    @property
    def previous(self) -> HistoryRecord | None:
        """Return the second-to-last record, or ``None``."""
        return self._records[-2] if len(self._records) >= 2 else None

    def last_n(self, n: int) -> list[HistoryRecord]:
        """Return the last *n* records (newest last).

        Args:
            n: Number of records to retrieve.

        Returns:
            List of up to *n* records.
        """
        return self._records[-n:]

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Export the full history as a DataFrame.

        Returns:
            DataFrame with one row per record.
        """
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self._records])

    def save_csv(self, path: Path) -> Path:
        """Persist the history to a CSV file.

        Args:
            path: Target file path.

        Returns:
            The written path.
        """
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Regime history saved → {} ({} records)", path.name, len(df))
        return path

    def clear(self) -> None:
        """Remove all stored records."""
        self._records.clear()
        logger.debug("Regime history cleared")

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def regime_durations(self) -> dict[str, list[int]]:
        """Compute consecutive-bar durations for each regime.

        Returns:
            Dictionary mapping regime labels to lists of streak
            lengths (in bars).
        """
        if not self._records:
            return {}

        durations: dict[str, list[int]] = {}
        current_regime = self._records[0].regime
        streak = 1

        for record in self._records[1:]:
            if record.regime == current_regime:
                streak += 1
            else:
                durations.setdefault(current_regime, []).append(streak)
                current_regime = record.regime
                streak = 1

        durations.setdefault(current_regime, []).append(streak)
        return durations
