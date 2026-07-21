"""Signal logging and persistence.

Logs generated signals to disk for performance auditing, strategy
validation, and backtesting review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from signals.signal import SignalType, TradeSignal


class SignalLogger:
    """Persist and track generated trade signals.

    Args:
        log_dir: Directory where signal logs will be saved.
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._signals: list[TradeSignal] = []

    def log_signal(self, signal: TradeSignal) -> None:
        """Record a signal and write it to the daily log file.

        Args:
            signal: The generated ``TradeSignal``.
        """
        self._signals.append(signal)

        # Skip logging HOLD signals to disk to save space, unless needed.
        if signal.signal == SignalType.HOLD:
            logger.debug(
                "[{}] HOLD | Regime: {} | Reason: {}",
                signal.strategy,
                signal.regime,
                " | ".join(signal.reason),
            )
            return

        logger.info(
            "[{}] {} | Regime: {} | Entry: {:.2f} | Target: {:.2f} | "
            "Stop: {:.2f} | R:R: {:.2f} | Conf: {:.2f}",
            signal.strategy,
            signal.signal.value,
            signal.regime,
            signal.entry,
            signal.take_profit,
            signal.stop_loss,
            signal.risk_reward,
            signal.confidence,
        )

        date_str = signal.timestamp.strftime("%Y-%m-%d")
        file_path = self.log_dir / f"signals_{date_str}.jsonl"

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal.to_dict()) + "\n")

    def get_all_signals(self) -> list[TradeSignal]:
        """Return all signals held in memory.

        Returns:
            List of ``TradeSignal`` objects.
        """
        return list(self._signals)

    def to_dataframe(self) -> pd.DataFrame:
        """Export in-memory signals to a Pandas DataFrame.

        Returns:
            DataFrame containing all logged signals.
        """
        if not self._signals:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self._signals])
