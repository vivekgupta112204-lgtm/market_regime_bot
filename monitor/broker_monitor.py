"""Broker connection tracking."""

from __future__ import annotations
from enum import Enum
from loguru import logger
import asyncio

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class BrokerMonitor:
    """Tracks continuous connection state with the active Exchange/Broker."""
    
    def __init__(self, broker_name: str):
        self.broker_name = broker_name
        self.state = ConnectionState.DISCONNECTED
        self.last_ping: float = 0.0
        self.error_count: int = 0

    def update_state(self, new_state: ConnectionState, error_msg: str | None = None):
        """Transitions state, triggering logs on connectivity issues."""
        if self.state != new_state:
            logger.info(f"Broker [{self.broker_name}] state changed: {self.state.name} -> {new_state.name}")
            self.state = new_state
            if new_state == ConnectionState.ERROR:
                self.error_count += 1
                if error_msg:
                    logger.error(f"Broker error: {error_msg}")

    def is_healthy(self) -> bool:
        return self.state == ConnectionState.CONNECTED
