"""Job scheduling and execution automation."""

from __future__ import annotations
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

class BotScheduler:
    """Manages the daily lifecycle, downloading data and triggering pipelines."""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
    
    async def schedule_daily_ingestion(self):
        logger.info("Executing scheduled Daily Data Ingestion...")
        # Integrates with data_manager.py
    
    async def schedule_retrain_check(self):
        logger.info("Executing scheduled Regime Model Drift Check...")
        from mlops.drift_detector import check_drift
        await check_drift()
        
    async def send_daily_report(self):
        logger.info("Executing scheduled End-Of-Day Report Generation...")
        from alerts.alert_manager import AlertManager
        alert = AlertManager()
        await alert.broadcast("Daily End of Day Report Generated.", level="INFO")

    def start(self):
        """Starts the scheduler in the background event loop."""
        # Run daily at 16:30 (Market Close for example)
        self.scheduler.add_job(self.schedule_daily_ingestion, CronTrigger(hour=16, minute=30))
        # Check drift every Saturday at midnight
        self.scheduler.add_job(self.schedule_retrain_check, CronTrigger(day_of_week='sat', hour=0, minute=0))
        # Send daily reports
        self.scheduler.add_job(self.send_daily_report, CronTrigger(hour=17, minute=0))
        
        self.scheduler.start()
        logger.info("Automation Scheduler Started.")
