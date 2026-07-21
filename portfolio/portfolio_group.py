"""Macro Portfolio abstractions enabling nested constraints."""

class PortfolioGroup:
    """Allows logical separation of capital (e.g., Growth vs Yield)."""
    
    def __init__(self, name: str, base_currency: str = "USD"):
        self.name = name
        self.currency = base_currency
        self.active = True
        
    def get_status_info(self) -> dict:
        return {
            "name": self.name,
            "status": "ONLINE" if self.active else "HALTED"
        }
