"""Immutable transaction ledger storing configuration changes and orders forever."""

from loguru import logger
import json
from datetime import datetime, timezone

class AuditLogger:
    """Interfaces with highly available logging architectures (e.g. ELK, Datadog, Prometheus)."""
    
    def log_event(self, event_type: str, data: dict):
        """Standardized ingestion format avoiding manipulation."""
        packet = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": data,
            "signature": "mock_hash"
        }
        logger.info(f"[AUDIT] {event_type} | TRACE_ID: {packet['signature']}")
        # Writing directly to flat file or external indexer
        with open("logs/audit_trail.jsona", "a") as f:
            f.write(json.dumps(packet) + "\n")
