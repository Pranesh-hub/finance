import yfinance as yf
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Fetch Data
# ------------------------------
tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS']
start_date, end_date = "2018-01-01", "2023-01-01"

print("Fetching market data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# Handle both single-level and multi-level columns
if isinstance(data.columns, pd.MultiIndex):
    data = data['Close']
else:
    data = data

data = data.dropna()
returns = data.pct_change().dropna()

# ------------------------------
# 2️⃣ Portfolio Environment (SB3 Compatible)
# ------------------------------
class PortfolioEnv(gym.Env):
    def __init__(self, returns):
        super().__init__()
        self.returns = returns.values
        self.num_assets = self.returns.shape[1]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,))
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.returns[self.current_step]
        return obs, {}

    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights)

        reward = np.dot(self.returns[self.current_step], weights)
        self.current_step += 1

        terminated = self.current_step >= len(self.returns) - 1
        truncated = False
        obs = self.returns[self.current_step] if not terminated else np.zeros(self.num_assets)
        info = {}

        return obs, reward, terminated, truncated, info

# ------------------------------
# 3️⃣ Optuna Hyperparameter Optimization
# ------------------------------
def optimize_rl(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    n_steps = trial.suggest_int('n_steps', 128, 2048, log=True)

    env = DummyVecEnv([lambda: PortfolioEnv(returns)])
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=lr, gamma=gamma, n_steps=n_steps)
    model.learn(total_timesteps=5000)

    obs, _ = env.reset()
    total_reward = 0
    for _ in range(len(returns) - 1):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward

print("Running Optuna hyperparameter tuning...")
study = optuna.create_study(direction='maximize')
study.optimize(optimize_rl, n_trials=20)

best_params = study.best_params
print(f"\nBest Hyperparameters Found: {best_params}")

# ------------------------------
# 4️⃣ Final RL Training with Best Params
# ------------------------------
env = DummyVecEnv([lambda: PortfolioEnv(returns)])
model = PPO('MlpPolicy', env, verbose=1, **best_params)
model.learn(total_timesteps=10000)

# ------------------------------
# 5️⃣ Evaluate & Compute Sharpe
# ------------------------------
obs, _ = env.reset()
rewards = []

for _ in range(len(returns) - 1):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        break

final_reward = np.sum(rewards)
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
sharpe_ratio = (mean_reward / std_reward) * np.sqrt(252)

print(f"\nFinal Evaluated Reward after Optimization: {final_reward:.4f}")
print(f"Mean Daily Reward: {mean_reward:.6f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# ------------------------------
# 6️⃣ Visualization
# ------------------------------
cumulative_returns = np.cumprod(1 + np.array(rewards)) - 1
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns, label="RL Portfolio Growth", color="dodgerblue")
plt.title("RL-Optimized Portfolio Performance")
plt.xlabel("Days")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()
