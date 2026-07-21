"""System and hardware health monitoring."""

from __future__ import annotations
import psutil
from datetime import datetime, timezone
from pydantic import BaseModel

class HealthMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    uptime_seconds: float
    timestamp: datetime

class HealthMonitor:
    """Monitors CPU, Memory, and Disk usage via psutil."""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.process = psutil.Process()

    def get_health(self) -> HealthMetrics:
        """Samples hardware and process metrics."""
        return HealthMetrics(
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds(),
            timestamp=datetime.now(timezone.utc)
        )
