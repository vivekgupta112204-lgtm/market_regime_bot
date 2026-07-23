import os
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.environment import TradingEnv

def build_baseline_brain():
    print("Building baseline PPO model...")
    df = yf.download("AAPL", period="1mo", interval="1d", progress=False)
    
    # Initialize env
    env = DummyVecEnv([lambda: TradingEnv(data=df)])
    
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64)
    model.learn(total_timesteps=100)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_agent.zip")
    print("Saved models/ppo_agent.zip")

if __name__ == "__main__":
    build_baseline_brain()
