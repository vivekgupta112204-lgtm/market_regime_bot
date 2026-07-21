"""Global infrastructure monitoring status aggregators."""

from loguru import logger
import psutil

class InfrastructureMonitor:
    """Exposes total hardware footprint usage across the clusters."""
    
    def fetch_health_status(self) -> str:
        cpu = psutil.cpu_percent()
        logger.debug(f"Host CPU Usage: {cpu}%")
        return "GREEN" if cpu < 90 else "RED"
