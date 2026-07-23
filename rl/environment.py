"""Gymnasium-compliant Reinforcement Learning Environment for Stock Trading."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from loguru import logger

class TradingEnv(gym.Env):
    """Simulates market interaction for an RL PPO Agent."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.max_steps = len(self.data) - 1
        
        # Action Space: [-1, 1] representing -100% (Short) to 100% (Long) allocation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obv Space: [Returns, Volatility, Spread, PrevAction, CurrentRegime, BalanceRatio, NLP_Sentiment, L2_Depth] = Size 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        
        return self._get_obs(), {}
        
    def _get_obs(self) -> np.ndarray:
        # Calculate dynamic features based on current step
        # In a real pipeline, data dataframe already holds pre-calculated technicals
        if self.current_step >= len(self.data):
             return np.zeros(8, dtype=np.float32)
             
        row = self.data.iloc[self.current_step]
        
        # Mocking or extracting from DF
        ret = row.get("Returns", 0.0)
        vol = row.get("Volatility", 0.0)
        regime = row.get("Regime", 0)
        
        obs = np.array([
            ret, 
            vol,
            0.01, # Mock spread
            self.position,
            float(regime),
            self.net_worth / self.initial_balance,
            row.get("NLP_Sentiment", 0.0), # FinBert / GenAI integration
            row.get("L2_Imbalance", 1.0) # Microstructure limit wall density
        ], dtype=np.float32)
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
