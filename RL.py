import yfinance as yf

tickers = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
data = yf.download(tickers, start='2022-09-01', end='2025-09-30')['Adj Close']

import numpy as np

returns = data.pct_change().dropna()
normalized_data = (data - data.min()) / (data.max() - data.min())

import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.stock_owned = np.zeros(len(data.columns))
        self.total_value = initial_balance

        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Hold, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_owned = np.zeros(len(self.data.columns))
        self.total_value = self.initial_balance
        return self.data.iloc[self.current_step].values

    def step(self, action):
        prev_state = self.data.iloc[self.current_step].values
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = 0

        current_state = self.data.iloc[self.current_step].values
        reward = 0

        if action == 0:  # Buy
            self.balance -= np.sum(current_state)
            self.stock_owned += current_state
        elif action == 2:  # Sell
            self.balance += np.sum(current_state)
            self.stock_owned -= current_state

        self.total_value = self.balance + np.sum(self.stock_owned * current_state)
        done = self.current_step == len(self.data) - 1
        return current_state, reward, done, {}

    def render(self):
        profit = self.total_value - self.initial_balance
        print(f'Step: {self.current_step}, Total Value: {self.total_value}, Profit: {profit}')

import random

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        max_next_q = max([self.get_q_value(next_state, a) for a in range(self.action_space)])
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(tuple(state), action)] = new_q

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_space)]
            return np.argmax(q_values)

env = StockTradingEnv(normalized_data)
agent = QLearningAgent(action_space=env.action_space.n)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state

# Backtesting
test_data = yf.download(tickers, start='2025-10-01', end='2025-10-31')['Adj Close']
test_returns = test_data.pct_change().dropna()
test_normalized = (test_data - test_data.min()) / (test_data.max() - test_data.min())

env = StockTradingEnv(test_normalized)
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
