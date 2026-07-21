"""Universal abstract definition of diverse financial instruments."""

from typing import Dict, Any

class BaseAsset:
    def __init__(self, symbol: str, asset_class: str):
        self.symbol = symbol
        self.asset_class = asset_class

class AssetManager:
    """Central factory and cache maintaining definitions of active assets."""
    
    def __init__(self):
        self._assets: Dict[str, BaseAsset] = {}

    def register_asset(self, asset: BaseAsset):
        self._assets[asset.symbol] = asset
        
    def get_asset(self, symbol: str) -> BaseAsset | None:
        return self._assets.get(symbol)
