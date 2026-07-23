"""Proximal Policy Optimization (PPO) training algorithms."""

from loguru import logger
import pandas as pd
import yfinance as yf
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

from rl.environment import TradingEnv

def fetch_training_data() -> pd.DataFrame:
    """Fetches high-density historical data to train the agent."""
    # We grab QQQ as a proxy for the total market to train structural regimes
    logger.info("Fetching historical structural data for RL pre-training...")
    df = yf.download("QQQ", period="2y", interval="1h", progress=False)
    
    # Feature Engineering for RL State
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    df['Regime'] = np.where(df['Returns'] > 0, 1.0, 0.0) # Simplified mock HMM regime
    df = df.dropna()
    return df

class RLTrainer:
    """Stable-Baselines3 Orchestrator."""
    
    def train_model(self, total_timesteps: int = 20000):
        logger.info(f"Initiating RL Model training loop for {total_timesteps} timesteps.")
        df = fetch_training_data()
        
        env = DummyVecEnv([lambda: TradingEnv(data=df)])
        
        # PPO is the gold standard for continuous action spaces
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        model.learn(total_timesteps=total_timesteps)
        
        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_agent.zip")
        logger.success("RL PPO Training complete. Saved to models/ppo_agent.zip")

if __name__ == "__main__":
    trainer = RLTrainer()
    trainer.train_model()
