"""
Advanced Reward Engineering for Reinforcement Learning Policies.
Implements Sharpe-Ratio Optimization, Drawdown Penalties, and Kelly Criterion Position Sizing.
"""

import numpy as np
from collections import deque


class RewardEngine:
    """Stateful reward calculator tracking rolling performance metrics across episodes."""
    
    def __init__(self, window: int = 50):
        self.returns_history = deque(maxlen=window)
        self.peak_net_worth = 0.0
        self.max_drawdown_seen = 0.0
        self.consecutive_losses = 0
        self.trade_count = 0
    
    def calculate_reward(self, current_net_worth: float, prev_net_worth: float, position: float) -> float:
        """
        Multi-Objective Reward Function combining:
        1. Risk-Adjusted Returns (Differential Sharpe Ratio)
        2. Maximum Drawdown Penalty
        3. Position Sizing Regularization (Kelly-inspired)
        4. Win-Streak Momentum Bonus
        5. Overtrading Penalty
        """
        
        # ═══════════════════════════════════════════
        # COMPONENT 1: Differential Sharpe Ratio
        # ═══════════════════════════════════════════
        pnl = current_net_worth - prev_net_worth
        step_return = pnl / (prev_net_worth + 1e-10)
        self.returns_history.append(step_return)
        
        if len(self.returns_history) >= 10:
            returns_arr = np.array(self.returns_history)
            mean_r = np.mean(returns_arr)
            std_r = np.std(returns_arr) + 1e-10
            # Differential Sharpe: rewards consistency, not just magnitude
            sharpe_component = mean_r / std_r
        else:
            sharpe_component = step_return * 10  # Early episodes: raw PnL scaled
        
        # ═══════════════════════════════════════════
        # COMPONENT 2: Maximum Drawdown Penalty
        # ═══════════════════════════════════════════
        if current_net_worth > self.peak_net_worth:
            self.peak_net_worth = current_net_worth
        
        current_drawdown = (self.peak_net_worth - current_net_worth) / (self.peak_net_worth + 1e-10)
        self.max_drawdown_seen = max(self.max_drawdown_seen, current_drawdown)
        
        # Exponential drawdown penalty (small drawdowns = small penalty, large = devastating)
        drawdown_penalty = -2.0 * (current_drawdown ** 2)
        
        # ═══════════════════════════════════════════
        # COMPONENT 3: Kelly Criterion Position Sizing
        # ═══════════════════════════════════════════
        # Penalize extreme positions (>80% allocation) and reward moderate sizing
        abs_position = abs(position)
        if abs_position > 0.8:
            kelly_penalty = -0.5 * (abs_position - 0.8)  # Heavy penalty for over-leverage
        elif abs_position > 0.5:
            kelly_penalty = -0.1 * (abs_position - 0.5)  # Mild penalty
        else:
            kelly_penalty = 0.0  # Optimal zone
        
        # ═══════════════════════════════════════════
        # COMPONENT 4: Win-Streak Momentum Bonus
        # ═══════════════════════════════════════════
        if pnl > 0:
            self.consecutive_losses = 0
            streak_bonus = 0.05  # Small bonus for profitable step
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 5:
                streak_bonus = -0.3  # Heavy penalty for 5+ consecutive losses (tilt detection)
            elif self.consecutive_losses >= 3:
                streak_bonus = -0.1
            else:
                streak_bonus = 0.0
        
        # ═══════════════════════════════════════════
        # COMPONENT 5: Overtrading Frequency Penalty
        # ═══════════════════════════════════════════
        self.trade_count += 1 if abs_position > 0.01 else 0
        overtrading_penalty = 0.0
        if self.trade_count > 200:
            overtrading_penalty = -0.001 * (self.trade_count - 200)
        
        # ═══════════════════════════════════════════
        # FINAL COMPOSITE REWARD
        # ═══════════════════════════════════════════
        total_reward = (
            sharpe_component * 1.0 +      # Primary: Risk-adjusted returns
            drawdown_penalty * 0.5 +       # Secondary: Capital preservation
            kelly_penalty * 0.3 +          # Tertiary: Position sizing discipline
            streak_bonus * 0.2 +           # Quaternary: Behavioral regularization
            overtrading_penalty            # Quinary: Frequency control
        )
        
        return float(np.clip(total_reward, -5.0, 5.0))


# Backward-compatible wrapper for existing code
_engine = RewardEngine()

def calculate_reward(current_net_worth: float, prev_net_worth: float, position: float) -> float:
    """Legacy-compatible function that routes to the advanced RewardEngine."""
    return _engine.calculate_reward(current_net_worth, prev_net_worth, position)
