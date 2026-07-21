"""Derivative constraint handler for FX Pairs."""

from multi_asset.asset_manager import BaseAsset

class ForexAsset(BaseAsset):
    def __init__(self, symbol: str, quote_currency: str, base_currency: str, pip_value: float = 0.0001):
        super().__init__(symbol, "Forex")
        self.quote = quote_currency
        self.base = base_currency
        self.pip_value = pip_value
