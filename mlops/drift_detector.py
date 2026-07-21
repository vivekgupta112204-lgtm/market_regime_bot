"""Statistical concept drift testing on incoming market data."""

import numpy as np
from loguru import logger
from typing import List

class DriftDetector:
    """Monitors incoming data streams for distribution shifts compared to training data."""
    
    def __init__(self, baseline_means: List[float], baseline_stds: List[float]):
        self.baseline_means = np.array(baseline_means)
        self.baseline_stds = np.array(baseline_stds)

    def is_drifting(self, recent_data: np.ndarray, z_score_threshold: float = 3.0) -> bool:
        """
        Uses a Z-score heuristic to identify if recent out-of-sample data
        is statistically deviating from the training distribution.
        """
        if len(recent_data) == 0:
            return False
            
        recent_means = np.mean(recent_data, axis=0)
        
        # Calculate Z-Scores mapping recent feature means to baseline distribution
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.abs(recent_means - self.baseline_means) / self.baseline_stds
        
        # If any feature strays beyond the threshold, flag drift
        max_deviation = np.nanmax(z_scores)
        if max_deviation > z_score_threshold:
            logger.warning(f"Data drift detected! Max feature Z-score: {max_deviation:.2f}")
            return True
            
        return False

async def check_drift():
    """Entry point for the apscheduler job."""
    detector = DriftDetector(baseline_means=[0.05, 1.2], baseline_stds=[0.1, 0.4])
    # Mock fetching recent data
    dummy_recent_data = np.random.normal(0.06, 1.5, (100, 2))
    
    if detector.is_drifting(dummy_recent_data):
        from automation.retrainer import AutoRetrainer
        retrainer = AutoRetrainer()
        await retrainer.trigger_retraining(trigger_reason="Statistical Feature Drift Detected")
