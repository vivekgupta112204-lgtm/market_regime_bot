"""Macro-Economic Filter (Federal Reserve Calendar & High Impact News)."""

import requests
from datetime import datetime, timezone
import pytz
from loguru import logger

class MacroAgent:
    """Detects heavy macroeconomic events (FOMC, CPI, NFP) to freeze trading."""
    
    def __init__(self):
        # Open source global economic JSON feed
        self.cal_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        
    def check_for_hurricane(self) -> bool:
        """Returns True if a 'High' impact USD event is happening *today*."""
        logger.info("MacroAgent scanning Global Economic Calendar for incoming storms...")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(self.cal_url, headers=headers, timeout=5)
            
            if res.status_code != 200:
                logger.warning(f"Failed to fetch calendar (Status: {res.status_code}). Continuing normal trading.")
                return False
                
            events = res.json()
            # Fetch today's date in string format matching the JSON structure (YYYY-MM-DD)
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            for ev in events:
                if ev.get("country") == "USD" and ev.get("impact") == "High":
                    ev_date_str = ev.get("date", "")[:10]  # Extract just the date part
                    if ev_date_str == today_str:
                        title = ev.get("title", "Unknown Macro Event")
                        logger.error(f"🚨 MACRO HURRICANE DETECTED TODAY: {title}. FED/CPI activity incoming.")
                        return True
                        
            logger.info("Macro Weather is clear. No FED actions today.")
            return False
            
        except Exception as e:
            logger.error(f"MacroAgent JSON parsing failed: {e}. Defaulting to safe (trading allowed).")
            return False
