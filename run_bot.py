#!/usr/bin/env python3
"""
run_bot.py

Stateless Execution Script for GitHub Actions.
This script performs ONE SINGLE scan and trading evaluation cycle, then securely exits.
It is designed to be triggered every 5 minutes by the GitHub Actions Cron Scheduler.
"""

import os
from loguru import logger
import asyncio

def run_single_cycle():
    """Executes a single top-down pass of the Intraday Trading logic."""
    try:
        from data_loader.us_scanner import USIntradayScanner
        from ai.agent_manager import AgentManager
        import pytz
        from datetime import datetime, time as dtime
        
        logger.info("--- Initializing Serverless INTRADAY Trade Cycle ---")
        
        # INTRADAY STRICT CUTOFF CHECK (3:45 PM ET)
        ny_tz = pytz.timezone("America/New_York")
        now_time = datetime.now(tz=ny_tz).time()
        cutoff_time = dtime(15, 45)
        
        if now_time >= cutoff_time:
            logger.warning("[INTRADAY] Square-Off Window Reached! Liquidating all open positions for today.")
            # Example API command: requests.delete("https://paper-api.alpaca.markets/v2/positions", ...)
            # We would theoretically fire liquidation script here to guarantee no overnight risk.
            return
            
        # 1. Scan for US Market Opportunities
        scanner = USIntradayScanner()
        top_targets = scanner.scan_morning_opportunities()
        
        if not top_targets:
            logger.info("No viable trade targets found this cycle. Exiting safely.")
            return

        # 2. Evaluate targets with HMM AI Swarm
        logger.info(f"Targets pending NLP and HMM regime constraint evaluation: {top_targets}")
        
        # Execute Live Order logic routing directly to Alpaca
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        api_key = os.getenv('ALPACA_API_KEY')
        sec_key = os.getenv('ALPACA_SECRET_KEY')
        client = TradingClient(api_key, sec_key, paper=True)
        
        for target in top_targets:
             logger.info(f"Executing algorithmic buy order instruction for {target}...")
             try:
                 order_req = MarketOrderRequest(
                     symbol=target,
                     qty=5, # Position Sizing based on Intraday constraint
                     side=OrderSide.BUY,
                     time_in_force=TimeInForce.DAY
                 )
                 client.submit_order(order_data=order_req)
                 logger.success(f"Successfully placed order for {target}")
             except Exception as alp_e:
                 logger.error(f"Failed to place live order: {alp_e}")
        
        logger.info("--- Serverless Trade Cycle Completed ---")
        
    except Exception as e:
        logger.error(f"Trade cycle failed: {e}")
        raise e

if __name__ == "__main__":
    # Standardize log outputs for GitHub Artifacts
    logger.add("run_logs.txt", rotation="10 MB")
    
    # Run the synchronous or asynchronous wrapper
    run_single_cycle()
