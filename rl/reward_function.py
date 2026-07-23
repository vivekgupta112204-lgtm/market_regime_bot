"""Dense and sparse reward shaping for reinforcement learning policies."""

def calculate_reward(current_net_worth: float, prev_net_worth: float, position: float) -> float:
    """Calculates continuous reward based on Portfolio Delta."""
    
    # Base reward is the change in portfolio value
    pnl = current_net_worth - prev_net_worth
    
    # Scale reward
    reward = pnl / current_net_worth if current_net_worth > 0 else 0
    
    # Simple risk penalty for high exposure (regularization)
    risk_penalty = 0.0001 * abs(position)
    
    return reward - risk_penalty
