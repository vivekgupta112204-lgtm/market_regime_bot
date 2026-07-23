"""
VIX Fear Index & Cross-Asset Correlation Defense Shield.
Monitors real-time market fear levels and detects structural correlation breakdowns.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from loguru import logger


class VIXDefenseShield:
    """Monitors the CBOE Volatility Index (VIX) and cross-asset correlation for systemic risk."""
    
    # VIX Thresholds (Industry Standard)
    VIX_CALM = 15.0        # Below 15 = Market is sleeping peacefully
    VIX_ELEVATED = 20.0    # 20-25 = Getting nervous
    VIX_FEAR = 25.0        # 25-30 = Institutional panic starting
    VIX_CRASH = 30.0       # 30+ = Full blown market crash / Black Swan territory
    
    def scan_fear_index(self) -> dict:
        """Reads real-time VIX level and returns risk classification."""
        logger.info("🌡️ VIX Defense Shield scanning market fear temperature...")
        
        try:
            vix_data = yf.download("^VIX", period="5d", interval="1h", progress=False)['Close']
            
            if vix_data.empty:
                return {"vix_level": 0.0, "regime": "UNKNOWN", "action": "PROCEED"}
            
            current_vix = float(vix_data.iloc[-1])
            prev_vix = float(vix_data.iloc[-2]) if len(vix_data) >= 2 else current_vix
            vix_spike = current_vix - prev_vix  # How fast fear is rising
            
            logger.info(f"📊 Current VIX: {current_vix:.2f} | 1H Change: {vix_spike:+.2f}")
            
            if current_vix >= self.VIX_CRASH:
                logger.critical(f"🔴 VIX CRASH ZONE ({current_vix:.1f}): FULL LOCKDOWN. No new positions authorized!")
                return {"vix_level": current_vix, "regime": "CRASH", "action": "LOCKDOWN"}
            
            elif current_vix >= self.VIX_FEAR:
                logger.error(f"🟠 VIX FEAR ZONE ({current_vix:.1f}): Reduce position sizes by 50%.")
                return {"vix_level": current_vix, "regime": "FEAR", "action": "HALF_SIZE"}
            
            elif current_vix >= self.VIX_ELEVATED:
                logger.warning(f"🟡 VIX ELEVATED ({current_vix:.1f}): Proceed with caution. Tighten stop-losses.")
                return {"vix_level": current_vix, "regime": "ELEVATED", "action": "CAUTION"}
            
            # VIX Spike Detection: If VIX jumped > 3 points in 1 hour, emergency brake
            elif vix_spike > 3.0:
                logger.critical(f"⚡ VIX SPIKE DETECTED (+{vix_spike:.1f} in 1H)! Emergency risk reduction!")
                return {"vix_level": current_vix, "regime": "SPIKE", "action": "HALF_SIZE"}
            
            else:
                logger.info(f"🟢 VIX CALM ({current_vix:.1f}): Market conditions optimal for execution.")
                return {"vix_level": current_vix, "regime": "CALM", "action": "PROCEED"}
                
        except Exception as e:
            logger.warning(f"VIX Defense Shield failed: {e}. Defaulting to PROCEED.")
            return {"vix_level": 0.0, "regime": "UNKNOWN", "action": "PROCEED"}

    def detect_correlation_breakdown(self) -> bool:
        """
        Detects if traditional market correlations are breaking down (crisis signal).
        Normal: SPY and QQQ move together (correlation > 0.85)
        Crisis: SPY and QQQ diverge (correlation drops below 0.6)
        """
        logger.info("🔗 Scanning cross-asset correlation matrix for structural breakdown...")
        
        try:
            data = yf.download(["SPY", "QQQ", "TLT", "GLD"], period="10d", interval="1h", progress=False)['Close']
            
            if data.empty or len(data) < 20:
                return False
            
            returns = data.pct_change().dropna()
            corr_matrix = returns.corr()
            
            spy_qqq_corr = float(corr_matrix.loc["SPY", "QQQ"])
            spy_tlt_corr = float(corr_matrix.loc["SPY", "TLT"])
            
            logger.info(f"📐 Correlation Matrix: SPY-QQQ={spy_qqq_corr:.3f} | SPY-TLT={spy_tlt_corr:.3f}")
            
            # Normal: SPY and QQQ highly correlated (>0.85), SPY and TLT negatively correlated
            # Crisis: Everything correlates to 1.0 (panic selling everything) or SPY-QQQ drops
            
            if spy_qqq_corr < 0.6:
                logger.critical("⚠️ CORRELATION BREAKDOWN: SPY-QQQ divergence detected! Sector rotation or crisis!")
                return True
            
            if spy_tlt_corr > 0.5:
                logger.critical("⚠️ CORRELATION ANOMALY: Stocks and Bonds moving TOGETHER (Flight to Cash)!")
                return True
            
            logger.info("🟢 Cross-asset correlations within normal structural bounds.")
            return False
            
        except Exception as e:
            logger.warning(f"Correlation scan failed: {e}")
            return False


class PortfolioExposureLimiter:
    """Prevents portfolio from exceeding maximum sector/total exposure limits."""
    
    SECTOR_MAP = {
        "TECH": ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "INTC", "ARM", "AVGO", "CRM", "CSCO", "ORCL", "IBM"],
        "FINANCE": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "V", "MA", "PYPL", "SQ"],
        "HEALTH": ["UNH", "JNJ", "PFE", "MRK", "ABT", "TMO", "ISRG", "SYK", "MDT", "ZTS", "BAX"],
        "ENERGY": ["XOM", "CVX"],
        "CRYPTO_PROXY": ["COIN", "MARA", "RIOT", "MSTR"],
        "CONSUMER": ["AMZN", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP", "DIS"],
        "MEME": ["GME", "AMC"],
    }
    
    MAX_SECTOR_CONCENTRATION = 0.40  # No single sector > 40% of portfolio
    MAX_SINGLE_STOCK = 0.15          # No single stock > 15% of portfolio
    
    def check_concentration_risk(self, current_positions: list, new_target: str) -> dict:
        """
        Validates if adding a new position would breach concentration limits.
        Returns: {"allowed": bool, "reason": str}
        """
        if not current_positions:
            return {"allowed": True, "reason": "Portfolio empty. Full allocation authorized."}
        
        # Find which sector the new target belongs to
        target_sector = "OTHER"
        for sector, symbols in self.SECTOR_MAP.items():
            if new_target in symbols:
                target_sector = sector
                break
        
        # Count existing sector exposure
        sector_count = {}
        for pos in current_positions:
            for sector, symbols in self.SECTOR_MAP.items():
                if pos in symbols:
                    sector_count[sector] = sector_count.get(sector, 0) + 1
        
        current_sector_weight = sector_count.get(target_sector, 0) / max(len(current_positions), 1)
        
        if current_sector_weight >= self.MAX_SECTOR_CONCENTRATION:
            reason = f"BLOCKED: {target_sector} sector already at {current_sector_weight*100:.0f}% concentration (Max: {self.MAX_SECTOR_CONCENTRATION*100:.0f}%)"
            logger.error(f"🚫 {reason}")
            return {"allowed": False, "reason": reason}
        
        # Check single-stock concentration
        stock_count = current_positions.count(new_target)
        stock_weight = stock_count / max(len(current_positions), 1)
        
        if stock_weight >= self.MAX_SINGLE_STOCK:
            reason = f"BLOCKED: {new_target} already at {stock_weight*100:.0f}% of portfolio (Max: {self.MAX_SINGLE_STOCK*100:.0f}%)"
            logger.error(f"🚫 {reason}")
            return {"allowed": False, "reason": reason}
        
        return {"allowed": True, "reason": f"Diversification check passed for {new_target} ({target_sector})."}
