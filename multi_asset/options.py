"""Derivative constraint handler for Options."""

from multi_asset.asset_manager import BaseAsset
from datetime import datetime

class OptionAsset(BaseAsset):
    def __init__(self, symbol: str, expiration_date: datetime, strike: float, option_type: str = "CALL", multiplier: int = 100):
        super().__init__(symbol, "Options")
        self.expiration = expiration_date
        self.strike = strike
        self.option_type = option_type
        self.multiplier = multiplier
