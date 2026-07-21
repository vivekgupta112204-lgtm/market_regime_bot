"""Example simulating live inputs wrapping the Paper environment yaml."""

from loguru import logger
import os

def paper_trade_spinup():
    os.environ["BOT_MODE"] = "PAPER"
    logger.info("Engaging Paper Trading Simulation Framework.")
    # Calls start_enterprise_platform() with PAPER configs
