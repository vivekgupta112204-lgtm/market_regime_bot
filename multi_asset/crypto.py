"""Derivative constraint handler for Digital Assets."""

from multi_asset.asset_manager import BaseAsset

class CryptoAsset(BaseAsset):
    def __init__(self, symbol: str, quote_currency: str = "USD"):
        super().__init__(symbol, "Crypto")
        self.quote_currency = quote_currency
        self.fractional_allowed = True
