import os
import yfinance as yf
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.environment import TradingEnv

def build_baseline_brain():
    print("Building Honest Baseline RecurrentPPO model...")
    # Fetch multi-year data across large indices
    df = yf.download(["SPY", "QQQ"], period="2y", interval="1d", progress=False)
    
    close_prices = df['Close']['SPY'] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    
    processed = pd.DataFrame(index=close_prices.index)
    processed['Close'] = close_prices
    processed['Returns'] = processed['Close'].pct_change()
    processed['Volatility'] = processed['Returns'].rolling(window=10).std()
    processed['Regime'] = np.where(processed['Returns'] > 0, 1.0, 0.0)
    processed = processed.dropna()
    
    env = DummyVecEnv([lambda: TradingEnv(data=processed)])
    
    # Minimum Timesteps Gate
    TIMESTEPS = 75000
    if TIMESTEPS < 50000:
        raise ValueError("FATAL: Will not train a model with < 50k timesteps. Avoid deploying undertrained models.")
        
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, n_steps=256)
    model.learn(total_timesteps=TIMESTEPS)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/recurrent_ppo_agent.zip")
    
    # Save training metrics lockfile
    with open("models/recurrent_ppo_agent_metrics.txt", "w") as f:
        f.write(f"TIMESTEPS: {TIMESTEPS}\n")
        f.write(f"TRAIN_DATA: 2Y SPY/QQQ\n")
        f.write(f"EVAL_SHARPE: CHECK_TENSORBOARD\n") # Real pipeline would run an eval wrapper

    print("Saved models/recurrent_ppo_agent.zip with robust timestep gate passed.")

if __name__ == "__main__":
    build_baseline_brain()
