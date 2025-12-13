# ============================================================
# RL PORTFOLIO OPTIMIZATION
# November Learning → December Robust Improvement
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
np.seterr(all="ignore")
# ------------------------------------------------------------
# 1️⃣ DATA FETCH
# ------------------------------------------------------------
tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS']
start_date, end_date = "2018-01-01", "2023-01-01"

print("Fetching market data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# Handle multi-index columns safely
if isinstance(data.columns, pd.MultiIndex):
    data = data['Close']

data = data.dropna()
returns = data.pct_change().dropna()

# ------------------------------------------------------------
# 2️⃣ TRAIN / TEST SPLIT (KEY CHANGE)
# ------------------------------------------------------------
train_returns = returns.loc[:'2022-10-31']
test_returns  = returns.loc['2022-11-01':'2022-12-31']

# ------------------------------------------------------------
# 3️⃣ IMPROVED PORTFOLIO ENVIRONMENT
# ------------------------------------------------------------
class PortfolioEnv(gym.Env):
    """
    Observation:
      [1-day returns | 5-day momentum | 20-day volatility]

    Reward:
      return - risk_penalty * volatility - turnover_penalty * turnover
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        window: int = 20,
        risk_penalty: float = 0.1,
        turnover_penalty: float = 0.001
    ):
        super().__init__()

        self.returns_df = returns
        self.returns = returns.values
        self.num_assets = self.returns.shape[1]
        self.window = window
        self.risk_penalty = risk_penalty
        self.turnover_penalty = turnover_penalty

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_assets * 3,),
            dtype=np.float32
        )

        self.current_step = self.window
        self.prev_weights = np.ones(self.num_assets) / self.num_assets

    def _get_obs(self):
        # 1-day returns
        ret_1 = self.returns[self.current_step]

        # 5-day momentum
        ret_mom = np.mean(
            self.returns[self.current_step-5:self.current_step],
            axis=0
        )

        # 20-day volatility
        vol = np.std(
            self.returns[self.current_step-self.window:self.current_step],
            axis=0
        )

        return np.concatenate([ret_1, ret_mom, vol]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.prev_weights = np.ones(self.num_assets) / self.num_assets
        return self._get_obs(), {}

    def step(self, action):
        # Normalize weights
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights) + 1e-8

        # Portfolio return
        portfolio_return = np.dot(
            self.returns[self.current_step],
            weights
        )

        # Portfolio volatility (rolling window)
        portfolio_vol = np.std(
            self.returns[self.current_step-self.window:self.current_step] @ weights
        )

        # Turnover penalty
        turnover = np.sum(np.abs(weights - self.prev_weights))

        # Risk-adjusted reward
        reward = (
            portfolio_return
            - self.risk_penalty * portfolio_vol
            - self.turnover_penalty * turnover
        )

        self.prev_weights = weights
        self.current_step += 1

        terminated = self.current_step >= len(self.returns) - 1
        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape,
            dtype=np.float32
        )

        return obs, reward, terminated, False, {}

# ------------------------------------------------------------
# 4️⃣ TRAIN PPO (STABLE CONFIG)
# ------------------------------------------------------------
train_env = DummyVecEnv([
    lambda: PortfolioEnv(train_returns)
])

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=512,
    batch_size=64,
    ent_coef=0.01,
    verbose=1
)

print("Training RL agent...")
model.learn(total_timesteps=30_000)

# ------------------------------------------------------------
# 5️⃣ OUT-OF-SAMPLE EVALUATION (NOV–DEC)
# ------------------------------------------------------------
test_env = DummyVecEnv([
    lambda: PortfolioEnv(test_returns)
])

obs = test_env.reset()
rewards = []
weights_hist = []

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)

    rewards.append(reward[0])
    weights_hist.append(action[0])

    if done[0]:
        break

rewards = np.array(rewards)
weights_hist = np.array(weights_hist)

# ------------------------------------------------------------
# 6️⃣ BENCHMARK: EQUAL WEIGHT
# ------------------------------------------------------------
eq_weights = np.ones(test_returns.shape[1]) / test_returns.shape[1]
eq_returns = test_returns.values @ eq_weights
eq_returns = eq_returns[:len(rewards)]

rl_cum = np.cumprod(1 + rewards) - 1
eq_cum = np.cumprod(1 + eq_returns) - 1

# ------------------------------------------------------------
# 7️⃣ METRICS & DIAGNOSTICS
# ------------------------------------------------------------
sharpe_rl = np.mean(rewards) / np.std(rewards) * np.sqrt(252)
sharpe_eq = np.mean(eq_returns) / np.std(eq_returns) * np.sqrt(252)

avg_max_weight = np.mean(np.max(weights_hist, axis=1))
turnover = np.mean(
    np.sum(np.abs(np.diff(weights_hist, axis=0)), axis=1)
)

print("\n========== NOV–DEC RESULTS ==========")
print(f"RL Sharpe Ratio       : {sharpe_rl:.3f}")
print(f"Equal-Weight Sharpe  : {sharpe_eq:.3f}")
print(f"Average Max Weight   : {avg_max_weight:.2f}")
print(f"Average Turnover     : {turnover:.3f}")

# ------------------------------------------------------------
# 8️⃣ VISUALIZATION
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(rl_cum, label="RL Portfolio", linewidth=2)
plt.plot(eq_cum, label="Equal Weight", linestyle="--")
plt.title("November–December Portfolio Performance")
plt.xlabel("Trading Days")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 9️⃣ SAVE DAILY DECEMBER WEIGHTS (CSV)
# ------------------------------------------------------------
weights_df = pd.DataFrame(
    weights_hist,
    columns=test_returns.columns,
    index=test_returns.index[:len(weights_hist)]
)

weights_df.to_csv("december_daily_weights.csv")

print("Saved: december_daily_weights.csv")