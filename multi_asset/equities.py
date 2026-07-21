"""Derivative constraint handler for Equities."""

from multi_asset.asset_manager import BaseAsset

class EquityAsset(BaseAsset):
    def __init__(self, symbol: str, margin_rate: float = 0.5):
        super().__init__(symbol, "Equities")
        self.margin_rate = margin_rate
