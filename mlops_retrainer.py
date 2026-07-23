"""
Self-Healing MLOps Retrainer Module.
Runs continuously on week-ends to ingest recent market drift and auto-tune the PPO Network weights.
"""

import os
import shutil
import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure we import the environment correctly
from rl.environment import TradingEnv

def fetch_drift_data() -> pd.DataFrame:
    """Fetches the latest 1-month of market data to capture recent economic drifts."""
    logger.info("📡 Fetching latest market topology for Neural Re-alignment...")
    # Train heavily on SPY & BTC to learn structural differences
    df = yf.download(["SPY", "QQQ"], period="1mo", interval="1h", progress=False)
    
    # We take the closing array of SPY for training proxy
    if 'Close' in df.columns:
        if isinstance(df['Close'], pd.DataFrame):
            close_prices = df['Close']['SPY']
        else:
            close_prices = df['Close']
    else:
        close_prices = df

    processed = pd.DataFrame(index=close_prices.index)
    processed['Close'] = close_prices
    processed['Returns'] = processed['Close'].pct_change()
    processed['Volatility'] = processed['Returns'].rolling(window=10).std()
    processed['Regime'] = np.where(processed['Returns'] > 0, 1.0, 0.0)
    
    return processed.dropna()

def auto_heal_model():
    logger.info("🤖 --- INITIATING WEEKEND SELF-HEALING MLOPS PROTOCOL ---")
    
    model_path = os.path.join("models", "ppo_agent.zip")
    backup_path = os.path.join("models", f"ppo_agent_backup_{datetime.now().strftime('%Y%m%d')}.zip")
    
    if not os.path.exists(model_path):
        logger.error("Core brain (ppo_agent.zip) missing. Cannot retrain.")
        return
        
    try:
        # Create a secure rollback backup
        shutil.copy(model_path, backup_path)
        logger.info(f"🛡️ Backed up generic weights to {backup_path}")
        
        # Load Recent Data
        df = fetch_drift_data()
        if len(df) < 50:
            logger.warning("Not enough live data to retrain this week.")
            return

        env = DummyVecEnv([lambda: TradingEnv(data=df)])
        
        # Load Existing Brain
        logger.info("🧠 Loading existing PPO Architecture...")
        model = PPO.load(model_path, env=env)
        
        # Incremental Continuous Learning (Does not wipe previous knowledge)
        logger.info("🔗 Injecting 10,000 new synaptic episodes into Neural Network...")
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        
        # Save overwritten new weights
        model.save(model_path)
        logger.success("✅ MLOps Retraining Complete. Model is now adapted to this week's market matrix!")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed during compilation: {e}")
        # Rollback feature
        if os.path.exists(backup_path):
            shutil.copy(backup_path, model_path)
            logger.info("↩️ Auto-Rolled back to previous stable brain weights.")

if __name__ == "__main__":
    auto_heal_model()
