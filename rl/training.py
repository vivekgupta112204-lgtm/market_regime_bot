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

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AntiOverfitFeatureExtractor(BaseFeaturesExtractor):
    """Custom Feature Extractor with Dropout for Regularization."""
    def __init__(self, observation_space, features_dim=64):
        super(AntiOverfitFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Adding Dropout (p=0.2) to prevent memorization of historical noise
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
    def forward(self, observations):
        return self.net(observations)


class RLTrainer:
    """Stable-Baselines3 Orchestrator."""
    
    def train_model(self, total_timesteps: int = 50000):
        logger.info(f"Initiating RL Model training loop for {total_timesteps} timesteps with Dropout Regularization.")
        df = fetch_training_data()
        
        env = DummyVecEnv([lambda: TradingEnv(data=df, is_training=True)])
        
        # Implement Agent Dropout Policy Kwargs
        policy_kwargs = dict(
            features_extractor_class=AntiOverfitFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
        
        # PPO is the gold standard for continuous action spaces
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0003, ent_coef=0.01)
        model.learn(total_timesteps=total_timesteps)
        
        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_agent.zip")
        logger.success("RL PPO Training complete. Saved to models/ppo_agent.zip")

if __name__ == "__main__":
    trainer = RLTrainer()
    trainer.train_model()
