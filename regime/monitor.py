"""Regime-change monitoring and alert generation.

Compares successive predictions and fires structured alerts whenever
the detected market regime transitions from one state to another.

Each alert contains:
* Old and new regime labels.
* Confidence of the new prediction.
* Timestamp of the transition.
* Human-readable reason text.

Alerts are logged via Loguru and collected in an in-memory list so
that downstream consumers (dashboards, webhooks, Telegram bots) can
process them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from regime.confidence import ConfidenceScore


@dataclass(frozen=True)
class RegimeAlert:
    """Structured alert for a regime change event.

    Attributes:
        timestamp: UTC datetime when the transition was detected.
        old_regime: The regime label before the change.
        new_regime: The regime label after the change.
        confidence: Posterior probability of the new regime.
        old_state_id: Numeric index of the old state.
        new_state_id: Numeric index of the new state.
        reason: Human-readable explanation.
    """

    timestamp: datetime
    old_regime: str
    new_regime: str
    confidence: float
    old_state_id: int
    new_state_id: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        Returns:
            Dict representation with ISO-formatted timestamp.
        """
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class RegimeMonitor:
    """Monitor predictions for regime transitions and emit alerts.

    The monitor maintains the *previous* regime so it can detect
    when a transition occurs.

    Args:
        min_confidence: Minimum confidence required to trigger an
            alert.  Transitions below this threshold are still logged
            at DEBUG level but not stored as alerts.
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        self._previous_state_id: int | None = None
        self._previous_regime: str | None = None
        self._alerts: list[RegimeAlert] = []
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def check(
        self,
        state_id: int,
        regime: str,
        confidence_score: ConfidenceScore,
        timestamp: datetime | None = None,
    ) -> RegimeAlert | None:
        """Compare the new prediction with the previous one.

        If the regime has changed, an alert is created, logged, and
        stored.

        Args:
            state_id: New predicted state index.
            regime: New regime label.
            confidence_score: Confidence scoring object.
            timestamp: Optional UTC timestamp (defaults to now).

        Returns:
            A ``RegimeAlert`` if a transition occurred, else ``None``.
        """
        ts = timestamp or datetime.now(timezone.utc)

        # First call — just record the baseline.
        if self._previous_state_id is None:
            self._previous_state_id = state_id
            self._previous_regime = regime
            logger.info(
                "Monitor initialised — baseline regime: {} (state={})",
                regime,
                state_id,
            )
            return None

        # No change.
        if state_id == self._previous_state_id:
            return None

        # Transition detected.
        confidence = confidence_score.dominant_confidence
        reason = (
            f"Regime transition: {self._previous_regime} → {regime} "
            f"(confidence={confidence:.2%})"
        )

        alert = RegimeAlert(
            timestamp=ts,
            old_regime=self._previous_regime or "Unknown",
            new_regime=regime,
            confidence=round(confidence, 4),
            old_state_id=self._previous_state_id,
            new_state_id=state_id,
            reason=reason,
        )

        if confidence >= self._min_confidence:
            self._alerts.append(alert)
            logger.warning(
                "⚡ REGIME CHANGE at {} — {} → {} (conf={:.2%})",
                ts.isoformat(),
                alert.old_regime,
                alert.new_regime,
                confidence,
            )
        else:
            logger.debug(
                "Low-confidence transition ignored: {} → {} (conf={:.2%})",
                alert.old_regime,
                alert.new_regime,
                confidence,
            )

        # Update baseline.
        self._previous_state_id = state_id
        self._previous_regime = regime

        return alert

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def alerts(self) -> list[RegimeAlert]:
        """All recorded alerts (oldest first)."""
        return list(self._alerts)

    @property
    def alert_count(self) -> int:
        """Number of recorded alerts."""
        return len(self._alerts)

    @property
    def latest_alert(self) -> RegimeAlert | None:
        """Most recent alert, or ``None``."""
        return self._alerts[-1] if self._alerts else None

    def clear_alerts(self) -> None:
        """Remove all stored alerts."""
        self._alerts.clear()
        logger.debug("Alert history cleared")
