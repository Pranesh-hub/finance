# ============================================================
# 1-MONTH PERFORMANCE EVALUATION
# RL-GENERATED DECEMBER PORTFOLIO
# ============================================================

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ LOAD RL-GENERATED PORTFOLIO CSV
# ------------------------------------------------------------
# Use ONE of the following:
# CSV_PATH = "december_final_portfolio.csv"
CSV_PATH = "december_daily_weights.csv"

df = pd.read_csv(CSV_PATH, index_col=0)

# If daily weights CSV → take last day as December portfolio
if df.shape[1] > 1:
    print("Detected daily weights CSV → using final December allocation")
    weights = df.iloc[-1]
else:
    weights = df.iloc[:, 0]

# Normalize weights (safety)
weights = weights / weights.sum()
tickers = weights.index.tolist()

print("\nDecember Portfolio Weights Used:")
print(weights)

# ------------------------------------------------------------
# 2️⃣ FETCH PRICE DATA (1-MONTH HORIZON)
# ------------------------------------------------------------
# Portfolio formed at start of December 2022
start_date = "2025-12-01"
end_date   = "2025-12-12"   # exactly 1 month

prices = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    progress=False
)["Close"].dropna()

# ------------------------------------------------------------
# 3️⃣ COMPUTE RETURNS
# ------------------------------------------------------------
returns = prices.pct_change().dropna()

# Individual stock cumulative returns
stock_cum_returns = (1 + returns).cumprod() - 1

# Portfolio cumulative returns
portfolio_returns = returns @ weights
portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1

# ------------------------------------------------------------
# 4️⃣ SAVE RESULTS TO CSV
# ------------------------------------------------------------
stock_cum_returns.to_csv("december_1m_stock_returns.csv")
portfolio_cum_returns.to_csv("december_1m_portfolio_return.csv")

summary = pd.DataFrame({
    "Weight": weights,
    "1M_Return": stock_cum_returns.iloc[-1]
})

summary["Weighted_Contribution"] = (
    summary["Weight"] * summary["1M_Return"]
)

summary.to_csv("december_1m_summary.csv")

print("\nSaved CSV files:")
print("december_1m_stock_returns.csv")
print("december_1m_portfolio_return.csv")
print("december_1m_summary.csv")

# ------------------------------------------------------------
# 5️⃣ PRINT SUMMARY METRICS
# ------------------------------------------------------------
portfolio_1m_return = portfolio_cum_returns.iloc[-1]

print("\n========== 1-MONTH RESULTS ==========")
print(f"Portfolio 1-Month Return: {portfolio_1m_return:.4%}")

print("\nStock Contributions:")
print(summary.sort_values("Weighted_Contribution", ascending=False))

# ------------------------------------------------------------
# 6️⃣ VISUALIZATION
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

for col in stock_cum_returns.columns:
    plt.plot(
        stock_cum_returns[col],
        alpha=0.5,
        label=col
    )

plt.plot(
    portfolio_cum_returns,
    color="black",
    linewidth=3,
    label="RL Portfolio"
)

plt.title("December Portfolio – 1 Month Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()
