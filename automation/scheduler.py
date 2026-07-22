"""Job scheduling and execution automation for US Swing Trading."""

from __future__ import annotations
import time
import pytz
from datetime import datetime, time as dtime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

class BotScheduler:
    """Manages the Swing Positional lifecycle and US market hour constraints."""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=pytz.timezone("America/New_York"))
    
    def _is_market_open(self) -> bool:
        now = datetime.now(pytz.timezone("America/New_York")).time()
        return dtime(9, 30) <= now <= dtime(16, 0)

    async def run_5min_trading_cycle(self):
        """Core logic executed every 1 hour during US hours for Swing evaluation."""
        if not self._is_market_open():
            return
            
        logger.info("[SWING] Executing Strategy Eval Cycle (Positions Held until Alpha reached).")
        # Submits jobs to AgentManager

    async def pre_market_scan(self):
        logger.info("[INTRADAY] Executing 09:15 AM Pre-Market US Scanner...")
        from data_loader.us_scanner import USIntradayScanner
        scanner = USIntradayScanner()
        top_stocks = scanner.scan_morning_opportunities()
        logger.success(f"Scan complete. Focusing on: {top_stocks}")
        
    async def schedule_retrain_check(self):
        logger.info("[INTRADAY] Executing scheduled Intraday Regime Drift Check...")
        from mlops.drift_detector import check_drift
        await check_drift()
        
    async def send_daily_report(self):
        logger.info("[INTRADAY] Executing 16:30 PM End-Of-Day Report Generation...")
        from alerts.alert_manager import AlertManager
        alert = AlertManager()
        await alert.broadcast("Daily Intraday Report Generated.", level="INFO")

    def start(self):
        """Starts the scheduler in the background event loop."""
        # 1. Scanner at 09:15 AM EST (Before 9:30 open)
        self.scheduler.add_job(self.pre_market_scan, CronTrigger(hour=9, minute=15, timezone="America/New_York"))
        
        # 2. Main engine evaluating every 5 minutes
        self.scheduler.add_job(self.run_5min_trading_cycle, CronTrigger(minute='*/5', timezone="America/New_York"))
        
        # 3. Model drift check daily after market
        self.scheduler.add_job(self.schedule_retrain_check, CronTrigger(hour=17, minute=0, timezone="America/New_York"))
        
        # 4. Reports after market
        self.scheduler.add_job(self.send_daily_report, CronTrigger(hour=16, minute=30, timezone="America/New_York"))
        
        self.scheduler.start()
        logger.info("US Intraday Automation Scheduler Started (EST Timezone strictly enforced).")
