"""Versioning controls for machine learning artifacts."""

from datetime import datetime, timezone
import hashlib

def generate_model_version(metrics: dict, architecture: str = "hmm_gaussian") -> str:
    """Generates a reproducible version hash based on architecture and timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    base_str = f"{architecture}_{timestamp}"
    
    # Simple short hash
    short_hash = hashlib.md5(base_str.encode()).hexdigest()[:6]
    return f"v_{timestamp}_{short_hash}"

def validate_model_promotion(current_metrics: dict, new_metrics: dict) -> bool:
    """Business logic defining if a model is 'better' and should enter production."""
    # Example logic: only promote if new model has higher log-likelihood or better out-of-sample sharpe
    if new_metrics.get("sharpe", 0) > current_metrics.get("sharpe", 0):
        return True
    return False
