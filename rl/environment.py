"""Gymnasium-compliant Reinforcement Learning Environment for Stock Trading."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from loguru import logger

class TradingEnv(gym.Env):
    """Simulates market interaction for an RL PPO Agent."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0, is_training: bool = True):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.max_steps = len(self.data) - 1
        self.is_training = is_training
        
        # Action Space: [-1, 1] representing -100% (Short) to 100% (Long) allocation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obv Space: [Returns, Volatility, Spread, PrevAction, CurrentRegime, BalanceRatio] + [PCA0...PCA3] = Size 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        
        return self._get_obs(), {}
        
    def _get_obs(self) -> np.ndarray:
        if self.current_step >= len(self.data):
             return np.zeros(10, dtype=np.float32)
             
        row = self.data.iloc[self.current_step]
        
        ret = float(row.get("Returns", 0.0))
        vol = float(row.get("Volatility", 0.0))
        regime = float(row.get("Regime", 0.0))
        
        pca_cols = [float(row.get(f"pca_{i}", 0.0)) for i in range(4)]
        
        obs = np.array([
            ret, 
            vol,
            0.01, # Mock spread
            self.position,
            regime,
            self.net_worth / self.initial_balance
        ] + pca_cols, dtype=np.float32)
        
        # Regularization: Synthetic Gaussian Noise Injection for anti-curve-fitting
        if self.is_training:
            noise = np.random.normal(loc=0.0, scale=0.005, size=obs.shape)
            obs = obs + noise
            
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advances simulator by computing PnL based on continuous action."""
        
        prev_net_worth = self.net_worth
        
        # Execute Action
        self.position = float(action[0])
        
        # Advance Step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if not done:
             # Calculate Returns
             row = self.data.iloc[self.current_step]
             ret = row.get("Returns", 0.0)
             # Basic PnL mapping: Return * Position Exposure
             pnl = self.position * ret * prev_net_worth
             self.net_worth += pnl
             
        from rl.reward_function import calculate_reward
        reward = calculate_reward(self.net_worth, prev_net_worth, self.position)
        
        truncated = False
        info = {"net_worth": self.net_worth}
        return self._get_obs(), float(reward), done, truncated, info
