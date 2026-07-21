"""Gymnasium-compliant Reinforcement Learning Environment for HMM Agent optimization."""

import numpy as np
from loguru import logger

class TradingEnv:
    """Simulates market interaction for an RL PPO Agent utilizing Regime states."""
    
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.current_step = 0
        self.max_steps = 1000
        
        # Actions: 0 = SHORT, 1 = CASH, 2 = LONG
        self.action_space = [0, 1, 2]
        
        # State: Returns, Volatility, Spread, PrevAction, CurrentRegime (0,1,2,3)
        self.observation_space = np.zeros(5)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        return np.zeros(5)
        
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Steps forward pushing action onto the simulator."""
        self.current_step += 1
        
        # MOCK STEP
        next_obs = np.random.normal(0, 1, 5)
        done = self.current_step >= self.max_steps
        
        from rl.reward_function import calculate_reward
        reward = calculate_reward(action, next_obs)
        
        return next_obs, reward, done, {}
