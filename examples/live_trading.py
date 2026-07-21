"""Example simulating live physical money risk wrapping the Production environment yaml."""

from loguru import logger
import os

def physical_trade_spinup():
    os.environ["BOT_MODE"] = "PRODUCTION"
    logger.critical("ENGAGING LIVE TRADING. PHYSICAL MONEY EXPOSED.")
    # Calls start_enterprise_platform() with PROD configs
