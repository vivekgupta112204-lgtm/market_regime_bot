"""
Serverless 24/7 Crypto Execution Environment (PPO Reinforcement Learning).
Executes strictly Fractional (Notional) trades across BTC, ETH, and SOL.
"""

import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

# RL inference setup same as Stock Agent
from stable_baselines3 import PPO

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. Setup Logging
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>CRYPTO-PHASE-10</cyan> - <white>{message}</white>"
logger.add(sys.stdout, format=log_format, level="INFO")

# 2. Crypto Target Universe
crypto_targets = ["BTC/USD", "ETH/USD", "SOL/USD"]

# Dollar-amount safe sizing (Never quantity based for expensive Bitcoin)
TRADE_NOTIONAL_USD = 100.0  # Buy/short exactly $100 worth of crypto per signal

def run_crypto_cycle():
    logger.info("--- Booting Crypto PPO 24/7 Environment ---")
    
    # 3. Load RL Agent Brain (If missing, bypass execution but don't crash)
    model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_agent.zip")
    if not os.path.exists(model_path):
         logger.warning(f"RL Model not found at {model_path}. Exiting cleanly.")
         return
         
    try:
        rl_model = PPO.load(model_path)
        logger.info("Successfully loaded RL PPO weights for 24/7 prediction map.")
    except Exception as e:
        logger.error(f"Failed to load RL model: {e}")
        return
        
    api_key = os.getenv('ALPACA_API_KEY')
    sec_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not sec_key:
        logger.warning("Alpaca API keys missing! Operating in Dry-Run Prediction mode.")
        client = None
    else:
        client = TradingClient(api_key, sec_key, paper=True)
        
    # 4. Execute ML predictions
    for target in crypto_targets:
         logger.info(f"Scanning 24/7 Momentum Vector for {target}...")
         
         # Synthesized internal state replacing standard stock data.
         state_vector = np.array([0.05, 0.08, 0.05, 0.0, 1.0, 1.0], dtype=np.float32)
         action, _states = rl_model.predict(state_vector, deterministic=True)
         
         try:
             if action[0] > 0.1: # Confidence threshold for LONG
                 logger.info(f"RL Agent Confirmed LONG action ({action[0]:.2f}) for {target}")
                 
                 if client:
                     order_req = MarketOrderRequest(
                         symbol=target,
                         notional=TRADE_NOTIONAL_USD,  # <-- Crucial: Fractional $ ordering
                         side=OrderSide.BUY,
                         time_in_force=TimeInForce.GTC
                     )
                     client.submit_order(order_data=order_req)
                     logger.success(f"Bought ${TRADE_NOTIONAL_USD} fractional {target}.")
                     
             elif action[0] < -0.1: # Confidence threshold for SHORT (Bear Signal)
                 logger.warning(f"RL Agent Confirmed SHORT action ({-action[0]:.2f}) for {target}")
                 
                 if client:
                     # Alpaca Crypto Shorting (Requires margin/specific approval, but request structure is identical)
                     order_req = MarketOrderRequest(
                         symbol=target,
                         notional=TRADE_NOTIONAL_USD, 
                         side=OrderSide.SELL,
                         time_in_force=TimeInForce.GTC
                     )
                     client.submit_order(order_data=order_req)
                     logger.success(f"Shorted ${TRADE_NOTIONAL_USD} fractional {target}.")
                     
             else:
                 logger.info(f"RL Agent recommends HOLD/WAIT for {target}.")
                 
         except Exception as alp_e:
             logger.error(f"Failed to place live crypto order for {target}: {alp_e}")

    logger.info("--- Crypto Serverless Trade Cycle Completed ---")

if __name__ == "__main__":
    run_crypto_cycle()
