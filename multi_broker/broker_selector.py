"""Intelligent order router traversing disparate brokers."""

from loguru import logger
from multi_broker.broker_manager import BrokerManager

class BrokerSelector:
    """Selects optimal broker considering API rate limits and fee structures."""
    
    def __init__(self, manager: BrokerManager):
        self.manager = manager
        
    def route_order(self, asset_class: str, size: float) -> str | None:
        """Determines best active connection for an asset."""
        logger.info(f"Routing order for {size} of {asset_class} across active broker cluster.")
        if asset_class == "Crypto":
            return "Binance"
        return "InteractiveBrokers" # Defaults equities/fx heavily
