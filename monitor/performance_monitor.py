"""Execution performance and latency tracking."""

from __future__ import annotations
import time
from typing import Dict
from loguru import logger
from collections import deque

class PerformanceMonitor:
    """Tracks latency for predictions, execution, and API responses."""
    
    def __init__(self, history_size: int = 100):
        self._metrics = {
            "prediction_latency_ms": deque(maxlen=history_size),
            "execution_latency_ms": deque(maxlen=history_size),
            "api_response_ms": deque(maxlen=history_size),
            "broker_latency_ms": deque(maxlen=history_size)
        }
        self.start_times: Dict[str, float] = {}

    def start_timer(self, key: str):
        self.start_times[key] = time.perf_counter()

    def stop_timer(self, metric: str, start_key: str):
        if start_key in self.start_times:
            elapsed = (time.perf_counter() - self.start_times.pop(start_key)) * 1000.0
            if metric in self._metrics:
                self._metrics[metric].append(elapsed)
            return elapsed
        return 0.0

    def get_averages(self) -> dict[str, float]:
        """Returns the rolling average latency for all tracked metrics."""
        averages = {}
        for m, q in self._metrics.items():
            averages[m] = sum(q) / len(q) if q else 0.0
        return averages
