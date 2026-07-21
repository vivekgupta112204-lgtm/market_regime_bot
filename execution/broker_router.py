"""Broker Router.

Instantiates and routes execution requests to the correct broker interface.
"""

from __future__ import annotations

from config.settings import Settings
from broker.broker_interface import BrokerInterface
from broker.paper_broker import PaperBroker
from broker.alpaca_broker import AlpacaBroker
from broker.binance_broker import BinanceBroker
from broker.bybit_broker import BybitBroker
from config.constants import Broker


class BrokerRouter:
    """Factory and router for broker integrations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._broker_instance: BrokerInterface | None = None

    def get_broker(self) -> BrokerInterface:
        """Get or instantiate the appropriate broker based on configuration."""
        if self._broker_instance is not None:
            return self._broker_instance

        if self.settings.execution.paper_mode:
            self._broker_instance = PaperBroker(
                self.settings.execution,
                initial_capital=self.settings.initial_capital
            )
        else:
            if self.settings.broker == Broker.ALPACA:
                self._broker_instance = AlpacaBroker(self.settings.api_keys)
            elif self.settings.broker == Broker.BINANCE:
                self._broker_instance = BinanceBroker(self.settings.api_keys)
            elif self.settings.broker == Broker.BYBIT:
                self._broker_instance = BybitBroker(self.settings.api_keys)
            else:
                # Fallback to PaperBroker if unsupported live broker
                self._broker_instance = PaperBroker(self.settings.execution, self.settings.initial_capital)

        self._broker_instance.connect()
        return self._broker_instance
