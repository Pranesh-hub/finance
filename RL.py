import yfinance as yf
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Fetch data
# ------------------------------
tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'ICICIBANK.NS']
data = yf.download(tickers, start='2022-09-01', end='2025-09-30')['Close'].ffill().values

# ------------------------------
# 2️⃣ Environment
# ------------------------------
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1_000_000, transaction_cost=0.001, slippage=0.001):
        super().__init__()
        self.df = df
        self.n_stock = df.shape[1]
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Action: [trade_0,...,trade_n, stop_loss_0,...,stop_loss_n] all in [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_stock*2,), dtype=np.float32)

        # Observation: normalized prices + stock_owned + balance
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_stock*2 + 1,), dtype=np.float32
        )

    def reset(self, seed=None, **kwargs):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_owned = np.zeros(self.n_stock)
        self.buy_price = np.zeros(self.n_stock)
        self.stop_losses = np.zeros(self.n_stock)
        self.total_value = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        norm_prices = self.df[self.current_step] / (self.df[0].max() + 1e-8)
        norm_stock = self.stock_owned / 1e3
        norm_balance = np.array([self.balance / self.initial_balance])
        return np.concatenate([norm_prices, norm_stock, norm_balance]).astype(np.float32)

    def step(self, action):
        prices = np.maximum(self.df[self.current_step], 1e-2)
        effective_prices = prices * (1 + np.random.normal(0, self.slippage, size=self.n_stock))
        prev_value = self.total_value
        portfolio_value = self.balance + np.sum(self.stock_owned * effective_prices)

        # Split action
        trade_actions = action[:self.n_stock]
        stop_loss_actions = action[self.n_stock:]

        for i in range(self.n_stock):
            # Map trade action [0,1] -> 0=Sell,1=Hold,2=Buy
            if trade_actions[i] < 0.33:
                act = 0
            elif trade_actions[i] < 0.66:
                act = 1
            else:
                act = 2

            # Adaptive stop-loss 1-20%
            self.stop_losses[i] = 0.01 + stop_loss_actions[i]*0.19

            # Trigger stop-loss
            if self.stock_owned[i] > 0 and prices[i] <= self.buy_price[i] * (1 - self.stop_losses[i]):
                self.balance += self.stock_owned[i] * effective_prices[i] * (1 - self.transaction_cost)
                self.stock_owned[i] = 0
                self.buy_price[i] = 0
                continue

            # Execute trade
            if act == 0:  # Sell
                self.balance += self.stock_owned[i] * effective_prices[i] * (1 - self.transaction_cost)
                self.stock_owned[i] = 0
                self.buy_price[i] = 0
            elif act == 2:  # Buy
                cash_to_use = self.balance * 1.0  # 100% allowed
                shares_to_buy = cash_to_use / effective_prices[i]
                self.stock_owned[i] += shares_to_buy
                self.balance -= shares_to_buy * effective_prices[i] * (1 + self.transaction_cost)
                self.buy_price[i] = effective_prices[i]

        # Update total value
        self.total_value = self.balance + np.sum(self.stock_owned * effective_prices)
        portfolio_vol = np.std(self.stock_owned * effective_prices) + 1e-6

        # Risk-adjusted reward
        reward = float(1000*((self.total_value - prev_value)/self.initial_balance) - 50*portfolio_vol/self.initial_balance)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}

# ------------------------------
# 3️⃣ Train PPO
# ------------------------------
env = DummyVecEnv([lambda: StockTradingEnv(data)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# ------------------------------
# 4️⃣ Backtest October 2025
# ------------------------------
test_data = yf.download(tickers, start='2025-10-01', end='2025-10-31')['Close'].ffill().values
test_env = StockTradingEnv(test_data)

obs, _ = test_env.reset()
values = []

for step in range(len(test_data)):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = test_env.step(action)
    values.append(test_env.total_value)
    if done:
        break

# ------------------------------
# 5️⃣ Plot portfolio value
# ------------------------------
plt.figure(figsize=(10,6))
plt.plot(values, label='Portfolio Value')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('RL Portfolio Backtest with Adaptive Stop-Loss - October 2025')
plt.legend()
plt.show()
