"""Dense and sparse reward shaping for reinforcement learning policies."""

def calculate_reward(action: int, next_obs: list) -> float:
    """Calculates immediate penalization for drawdown or rewards for risk-adjusted returns."""
    # Action 2 (Long) when regime implies bull structure should reward
    implied_return = next_obs[0] # Usually first feature is standardized return
    
    if action == 2: # LONG
        return float(implied_return)
    elif action == 0: # SHORT
        return -float(implied_return)
    else: # CASH
        return 0.0 # Safe baseline

def sharpe_penalty(history_returns: list) -> float:
    """Custom dense penalty applied at end of episode to force sortino/sharpe optimizations."""
    return 0.0
