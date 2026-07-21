"""Derivative constraint handler for Futures."""

from multi_asset.asset_manager import BaseAsset
from datetime import datetime

class FutureAsset(BaseAsset):
    def __init__(self, symbol: str, expiration_date: datetime, multiplier: float):
        super().__init__(symbol, "Futures")
        self.expiration = expiration_date
        self.multiplier = multiplier
