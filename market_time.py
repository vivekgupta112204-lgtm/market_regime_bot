#!/usr/bin/env python3
"""
market_time.py

This script determines if the US Stock Market (NYSE/NASDAQ) is currently open.
It uses Python's built-in zoneinfo module to correctly handle Daylight Saving Time (DST)
without hardcoding UTC or IST offsets.

Market Hours: Monday - Friday, 9:30 AM ET to 4:00 PM ET.
"""

import sys
from datetime import datetime, time
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for Python versions < 3.9 if needed, though Actions uses 3.12
    from backports.zoneinfo import ZoneInfo

def is_market_open() -> bool:
    """
    Checks if the current time falls within US Market hours.
    Returns: True if open, False if closed.
    """
    # 1. Get current time in America/New_York timezone (Handles DST automatically)
    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(tz=ny_tz)
    
    # 2. Check if it's a weekend (Monday=0, Sunday=6)
    if now_ny.weekday() >= 5:
        return False
        
    # 3. Check time bounds (9:30 AM to 4:00 PM)
    market_open_time = time(9, 30)
    market_close_time = time(16, 0)
    
    if market_open_time <= now_ny.time() <= market_close_time:
        return True
        
    return False

if __name__ == "__main__":
    if is_market_open():
        print("Market Open")
        sys.exit(0) # Exit code 0 implies success (Market is open)
    else:
        print("Market Closed")
        sys.exit(1) # Exit code 1 implies failure (Market is closed, stop CI/CD pipeline)
