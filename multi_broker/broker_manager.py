"""API connection manager spanning across disparate broker technologies."""

from loguru import logger
from typing import Dict, Any

class BrokerManager:
    """Manages active websockets and REST pools pointing at multiple external vendors."""
    
    def __init__(self):
        self._active_brokers: Dict[str, Any] = {}
        
    def register_broker(self, name: str, client_conn):
        self._active_brokers[name] = client_conn
        logger.info(f"Registered connection pool for broker: {name}")
        
    def get_broker(self, name: str):
        return self._active_brokers.get(name)
        
    def count_active(self) -> int:
        return len(self._active_brokers)
        
    # Mocking for Phase 11 validations
    def _mock_populate(self):
        self.register_broker("Alpaca", "MockConnA")
        self.register_broker("InteractiveBrokers", "MockConnB")
        self.register_broker("Binance", "MockConnC")
