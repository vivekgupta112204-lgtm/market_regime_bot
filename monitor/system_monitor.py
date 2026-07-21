"""Master Monitor Aggregator."""

from __future__ import annotations
from typing import Any
from monitor.health_monitor import HealthMonitor
from monitor.performance_monitor import PerformanceMonitor
from monitor.broker_monitor import BrokerMonitor, ConnectionState

class SystemMonitor:
    """Aggregates all monitoring subsystems into a single interface for the API."""
    
    def __init__(self, broker_name: str = "PaperBroker"):
        self.health = HealthMonitor()
        self.performance = PerformanceMonitor()
        self.broker = BrokerMonitor(broker_name)

    def get_system_status(self) -> dict[str, Any]:
        """Returns a snapshot of the entire system state suitable for the dashboard."""
        h_metrics = self.health.get_health()
        return {
            "status": "warning" if self.broker.state != ConnectionState.CONNECTED else "running",
            "health": {
                "cpu_percent": h_metrics.cpu_percent,
                "memory_percent": h_metrics.memory_percent,
                "disk_percent": h_metrics.disk_percent,
                "uptime_seconds": h_metrics.uptime_seconds
            },
            "performance": self.performance.get_averages(),
            "broker": {
                "name": self.broker.broker_name,
                "state": self.broker.state.value,
                "errors": self.broker.error_count
            }
        }
