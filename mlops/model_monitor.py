"""Gathers model tracking and lifecycle operations."""

from datetime import datetime
from loguru import logger
from mlops.model_registry import ModelRegistry

class ModelMonitor:
    """Supervises the live deployed model metrics relative to its test baseline."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
    def check_performance_degradation(self, current_live_sharpe: float) -> bool:
        """Compares current live trading metrics to the model's test-time metrics."""
        active = self.registry.get_active_model()
        if not active:
            return False
            
        baseline_sharpe = active.get("metrics", {}).get("sharpe", 0)
        
        # If live sharpe drops below 60% of test sharpe, it's considered degraded
        if current_live_sharpe < (baseline_sharpe * 0.6):
            logger.warning(f"Model degradation! Live Sharpe {current_live_sharpe:.2f} is massively underperforming test baseline {baseline_sharpe:.2f}.")
            return True
            
        return False
