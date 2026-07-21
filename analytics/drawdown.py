"""Drawdown analysis for portfolio analytics.

Computes maximum drawdown, drawdown duration, recovery time,
drawdown series, and underwater equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DrawdownEvent:
    """A single drawdown episode.

    Attributes:
        start: Start timestamp of the drawdown.
        trough: Timestamp of the deepest point.
        end: Timestamp when equity recovered (or None if ongoing).
        depth: Maximum depth of this drawdown as a positive percentage.
        duration: Total bars from start to recovery.
        recovery_bars: Bars from trough to recovery.
    """

    start: Any = None
    trough: Any = None
    end: Any = None
    depth: float = 0.0
    duration: int = 0
    recovery_bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "start": str(self.start) if self.start else "",
            "trough": str(self.trough) if self.trough else "",
            "end": str(self.end) if self.end else "",
            "depth": round(self.depth, 4),
            "duration": self.duration,
            "recovery_bars": self.recovery_bars,
        }


class DrawdownAnalyzer:
    """Analyse drawdowns in an equity curve.

    Args:
        equity_series: Time-indexed portfolio value series.
    """

    def __init__(self, equity_series: pd.Series) -> None:
        self._equity = equity_series.copy()
        self._drawdown_series: pd.Series | None = None
        self._events: list[DrawdownEvent] | None = None

    @property
    def drawdown_series(self) -> pd.Series:
        """Percentage drawdown at each point (positive = underwater).

        Returns:
            Series of drawdown percentages.
        """
        if self._drawdown_series is None:
            running_max = self._equity.cummax()
            self._drawdown_series = ((running_max - self._equity) / running_max) * 100.0
        return self._drawdown_series.copy()

    @property
    def underwater_curve(self) -> pd.Series:
        """Underwater equity curve (negative values when in drawdown).

        Returns:
            Series representing how far below the high-water mark.
        """
        return -self.drawdown_series

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a positive percentage.

        Returns:
            The deepest drawdown percentage.
        """
        dd = self.drawdown_series
        return float(dd.max()) if len(dd) > 0 else 0.0

    @property
    def max_drawdown_duration(self) -> int:
        """Longest drawdown duration in bars.

        Returns:
            Number of bars of the longest drawdown episode.
        """
        events = self.events
        if not events:
            return 0
        return max(e.duration for e in events)

    @property
    def events(self) -> list[DrawdownEvent]:
        """Identify all drawdown events.

        Returns:
            List of :class:`DrawdownEvent` instances.
        """
        if self._events is not None:
            return self._events

        dd = self.drawdown_series
        self._events = []

        if len(dd) == 0:
            return self._events

        in_drawdown = False
        current_event = DrawdownEvent()

        for i, (ts, val) in enumerate(dd.items()):
            if val > 0 and not in_drawdown:
                in_drawdown = True
                current_event = DrawdownEvent(start=ts, depth=val, trough=ts)
            elif val > 0 and in_drawdown:
                if val > current_event.depth:
                    current_event.depth = val
                    current_event.trough = ts
            elif val == 0 and in_drawdown:
                in_drawdown = False
                current_event.end = ts
                start_idx = dd.index.get_loc(current_event.start)
                end_idx = i
                current_event.duration = end_idx - start_idx

                trough_idx = dd.index.get_loc(current_event.trough)
                current_event.recovery_bars = end_idx - trough_idx

                self._events.append(current_event)

        if in_drawdown:
            start_idx = dd.index.get_loc(current_event.start)
            current_event.duration = len(dd) - start_idx
            trough_idx = dd.index.get_loc(current_event.trough)
            current_event.recovery_bars = len(dd) - trough_idx
            self._events.append(current_event)

        return self._events

    @property
    def average_drawdown(self) -> float:
        """Average drawdown depth across all events.

        Returns:
            Average drawdown as a positive percentage.
        """
        events = self.events
        if not events:
            return 0.0
        return sum(e.depth for e in events) / len(events)

    @property
    def average_recovery_time(self) -> float:
        """Average recovery time in bars.

        Returns:
            Average number of bars to recover from drawdowns.
        """
        recovered = [e for e in self.events if e.end is not None]
        if not recovered:
            return 0.0
        return sum(e.recovery_bars for e in recovered) / len(recovered)

    def top_drawdowns(self, n: int = 5) -> list[DrawdownEvent]:
        """Return the N deepest drawdowns.

        Args:
            n: Number of drawdowns to return.

        Returns:
            List of the deepest :class:`DrawdownEvent` instances.
        """
        events = sorted(self.events, key=lambda e: e.depth, reverse=True)
        return events[:n]

    def compute_all(self) -> dict[str, Any]:
        """Compute all drawdown metrics.

        Returns:
            Dictionary of drawdown analytics.
        """
        return {
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_duration": self.max_drawdown_duration,
            "average_drawdown": round(self.average_drawdown, 2),
            "average_recovery_time": round(self.average_recovery_time, 1),
            "total_drawdown_events": len(self.events),
            "top_5_drawdowns": [e.to_dict() for e in self.top_drawdowns(5)],
        }
