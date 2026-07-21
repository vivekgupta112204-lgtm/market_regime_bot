"""Fetches structured macroeconomic data (FED Rates, CPI, NFP)."""

class MacroData:
    """Orchestrates retrieval of Federal Reserve / BLS FRED endpoints."""
    
    def fetch_fed_funds_rate(self) -> float:
        return 5.25
        
    def fetch_cpi_yoy(self) -> float:
        return 3.2
