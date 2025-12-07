# stock_picker_max_sharpe.py
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier

# Optional sentiment libraries (used only if available)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# Optional Google Trends via pytrends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except Exception:
    PYTRENDS_AVAILABLE = False

# -----------------------
# TUNING KNOBS (editable)
# -----------------------
TICKERS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","TCS.NS","INFY.NS",
    "KOTAKBANK.NS","LT.NS","BHARTIARTL.NS","AXISBANK.NS","ITC.NS",
    "HINDUNILVR.NS","ASIANPAINT.NS","SUNPHARMA.NS","MARUTI.NS","BAJFINANCE.NS",
    "SBIN.NS","TITAN.NS","NTPC.NS","ONGC.NS","POWERGRID.NS"
]
START_DATE = "2024-10-01"
END_DATE = None   # None -> up to today
TOP_K = 30                    # number of stocks passed to optimizer (cap)
MAX_MISSING_FRAC = 0.1        # drop tickers with >10% missing values
SHORT_LT = 20                 # short lookback (days) for dynamic momentum
MEDIUM_LT = 60                # medium lookback (days)
RECENT_DAYS = 63              # recent window (approx 3 months)
ROLLING_SHARPE_WINDOW = 63    # window for rolling Sharpe (days)
BETA_THRESHOLD = 1.5          # drop tickers with beta > this
SENTIMENT_WEIGHT = 0.15       # how much sentiment contributes to final_score
TILT_STRENGTH = 0.12          # how much to tilt mu by final_score
RISK_FREE_RATE = 0.0          # used by optimizer if desired (PyPortfolioOpt uses 0 by default)
# Weights for combined quant signals (sum should be 1 before adding dynamic mom)
QUANT_WEIGHTS = {
    "momentum_vol": 0.4,
    "rolling_sharpe": 0.4,
    "beta_inv": 0.1,
    "drawdown_inv": 0.1
}
# -----------------------

def zscore(s: pd.Series) -> pd.Series:
    s2 = s.replace([np.inf, -np.inf], np.nan).fillna(0)
    std = s2.std()
    return (s2 - s2.mean()) / std if std and not np.isnan(std) else s2 * 0

# 1) Download price data
print("Downloading price data...")
if END_DATE:
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=True)
else:
    raw = yf.download(TICKERS, start=START_DATE, progress=True)

if "Close" in raw.columns:
    data = raw["Close"].copy()
else:
    # older yfinance versions may return single-level frame
    data = raw.copy()

# drop tickers with no data at all
data = data.dropna(axis=1, how="all")
# drop tickers with too many missing values
data = data.loc[:, data.isna().mean() < MAX_MISSING_FRAC]
# forward/backfill small gaps
data = data.fillna(method="ffill").fillna(method="bfill")

if data.shape[1] == 0:
    raise RuntimeError("No valid tickers after cleaning — check your ticker list and connection.")

daily_returns = data.pct_change().dropna()
print(f"Using {data.shape[1]} tickers for scoring.")

# 2) Compute base historical metrics (1-year / full window)
print("Computing base metrics (historical & recent)...")
mu_long = expected_returns.mean_historical_return(data, frequency=252)  # long-term (full window)
# recent 3-month mean return (annualized)
if len(data) >= RECENT_DAYS:
    mu_recent = expected_returns.mean_historical_return(data.last(f"{RECENT_DAYS}D"), frequency=252)
else:
    mu_recent = mu_long.copy()

vol = daily_returns.std() * np.sqrt(252)         # annualized volatility
# rolling Sharpe (compute rolling mean/std, take last available row)
if len(daily_returns) >= ROLLING_SHARPE_WINDOW:
    rm = daily_returns.rolling(window=ROLLING_SHARPE_WINDOW).mean()
    rs = daily_returns.rolling(window=ROLLING_SHARPE_WINDOW).std()
    rolling_sharpe = (rm / rs).iloc[-1] * np.sqrt(252)
else:
    rolling_sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

# dynamic momentum (short + medium blended)
if len(data) >= MEDIUM_LT:
    short_mom = data.pct_change(SHORT_LT).iloc[-1]
    medium_mom = data.pct_change(MEDIUM_LT).iloc[-1]
else:
    short_mom = data.pct_change(SHORT_LT).iloc[-1].fillna(0)
    medium_mom = data.pct_change(MEDIUM_LT).iloc[-1].fillna(0)
dynamic_mom = 0.6 * short_mom + 0.4 * medium_mom

# momentum-volatility hybrid over recent window
recent_ret = daily_returns.iloc[-RECENT_DAYS:] if len(daily_returns) >= RECENT_DAYS else daily_returns
mv_mean = recent_ret.mean()
mv_std = recent_ret.std().replace(0, np.nan)
momentum_vol_score = (mv_mean / mv_std).replace([np.inf, -np.inf], np.nan).fillna(0)

# max drawdown (negative number, more negative = worse)
cum = (1 + daily_returns).cumprod()
running_max = cum.cummax()
drawdown = cum / running_max - 1
max_dd = drawdown.min().fillna(0)

# 3) Beta vs Nifty (^NSEI)
print("Computing betas vs Nifty index (^NSEI)...")
try:
    idx_raw = yf.download("^NSEI", start=START_DATE, progress=False)
    if "Close" in idx_raw.columns:
        idx_series = idx_raw["Close"].copy()
    else:
        idx_series = idx_raw.copy()
    idx_ret = idx_series.pct_change().reindex(daily_returns.index).fillna(method="ffill").dropna()
    var_idx = idx_ret.var()
    betas = {}
    for col in daily_returns.columns:
        aligned = pd.concat([daily_returns[col].reindex(idx_ret.index).dropna(), idx_ret], axis=1).dropna()
        if aligned.shape[0] < 2:
            betas[col] = np.nan
            continue
        cov = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])[0,1]
        betas[col] = cov / var_idx if var_idx>0 else np.nan
    betas = pd.Series(betas).fillna(0)
except Exception as e:
    print("Warning: failed to compute betas:", e)
    betas = pd.Series(0, index=daily_returns.columns)

# beta inverse score
beta_inv = 1 / (1 + betas.clip(lower=0))   # large beta -> small score

# 4) Sentiment (best-effort): VADER headlines + Google Trends (if available)
sentiment = pd.Series(0.0, index=data.columns)
if VADER_AVAILABLE:
    print("Computing VADER/TextBlob sentiment (best-effort headlines)...")
    sid = SentimentIntensityAnalyzer()
    for t in data.columns:
        try:
            ticker_obj = yf.Ticker(t)
            news = getattr(ticker_obj, "news", []) or []
            if not news:
                # fallback: use company name token
                headlines = [t.split(".NS")[0] + " recent news"]
            else:
                headlines = [n.get("title","") for n in news[:6]]
            scores = []
            for h in headlines:
                if not h:
                    continue
                s_v = sid.polarity_scores(h).get("compound", 0.0)
                s_tb = TextBlob(h).sentiment.polarity
                scores.append(0.5*(s_v + s_tb))
            if scores:
                sentiment.loc[t] = np.mean(scores)
        except Exception:
            sentiment.loc[t] = 0.0

else:
    print("VADER/TextBlob unavailable — skipping news sentiment.")

if PYTRENDS_AVAILABLE:
    try:
        print("Computing Google Trends interest (batch queries) — may be slow...")
        pytrends = TrendReq(hl='en-US', tz=330)
        # query by name tokens in small batches
        names = [t.split(".NS")[0] for t in data.columns]
        for i in range(0, len(names), 5):
            batch = names[i:i+5]
            pytrends.build_payload(batch, timeframe='today 3-m')
            df = pytrends.interest_over_time()
            if df.empty:
                continue
            for nm in batch:
                if nm in df.columns:
                    vals = df[nm].replace(0, np.nan).dropna()
                    if len(vals) >= 2:
                        change = (vals.iloc[-1] / vals.iloc[0]) - 1
                        sentiment.loc[f"{nm}.NS"] += np.tanh(change)
    except Exception as e:
        print("Google Trends fetch failed or rate-limited:", e)
else:
    print("pytrends unavailable — skipping Google Trends.")

# Normalize sentiment to [-1,1] roughly
if sentiment.abs().max() != 0:
    sentiment = sentiment / (1 + sentiment.abs().max())

# 5) Normalize and combine quant signals
print("Combining quant signals into a final score...")
z_mv = zscore(momentum_vol_score)
z_rs = zscore(rolling_sharpe)
z_beta = zscore(beta_inv)
z_dd = zscore(-max_dd)   # invert drawdown (more negative dd => worse; invert to make higher better)
z_dynmom = zscore(dynamic_mom)

quant_combined = (
    QUANT_WEIGHTS["momentum_vol"] * z_mv +
    QUANT_WEIGHTS["rolling_sharpe"] * z_rs +
    QUANT_WEIGHTS["beta_inv"] * z_beta +
    QUANT_WEIGHTS["drawdown_inv"] * z_dd
)

# Blend dynamic momentum into quant combined (0.3 dynamic)
quant_with_dynamic = 0.7 * quant_combined + 0.3 * z_dynmom

# final score = (1 - SENTIMENT_WEIGHT) * quant + SENTIMENT_WEIGHT * sentiment_z
z_sent = zscore(sentiment)
final_score = (1 - SENTIMENT_WEIGHT) * zscore(quant_with_dynamic) + SENTIMENT_WEIGHT * z_sent
final_score = final_score.dropna()

# Filter by beta threshold
filtered = final_score.index[betas.reindex(final_score.index).fillna(0).abs() <= BETA_THRESHOLD]
final_score = final_score.reindex(filtered).dropna()

# If empty, fall back to top by recent mu
if final_score.empty:
    print("Warning: final_score empty after filters. Falling back to top historical returns.")
    fallback = mu_recent.sort_values(ascending=False).head(min(TOP_K, len(mu_recent))).index.tolist()
    topk = fallback
else:
    topk = final_score.sort_values(ascending=False).head(TOP_K).index.tolist()

print(f"Selected {len(topk)} tickers for optimization (top_k={TOP_K}):")
print(topk)

# 6) Build mu and covariance for selected universe
sub = data[topk].dropna(axis=1, how="all")
if sub.shape[1] == 0:
    raise RuntimeError("No valid data for selected tickers.")

recent_mu = expected_returns.mean_historical_return(sub.last(f"{RECENT_DAYS}D"), frequency=252) \
            if len(sub) >= RECENT_DAYS else expected_returns.mean_historical_return(sub, frequency=252)
long_mu = expected_returns.mean_historical_return(sub, frequency=252)
mu = 0.65 * recent_mu + 0.35 * long_mu   # dynamic blend favouring recent but anchored by long term

# tilt mu slightly toward final_score (z-scored)
tilt = final_score.reindex(mu.index).fillna(0)
mu = mu * (1 + TILT_STRENGTH * zscore(tilt))

# covariance with shrinkage
S = risk_models.CovarianceShrinkage(sub).ledoit_wolf()

# 7) Optimize (max Sharpe)
print("Running max-Sharpe optimization...")
try:
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    weights = ef.clean_weights()
    portfolio = pd.DataFrame(list(weights.items()), columns=["Stock", "Weight"])
    portfolio = portfolio[portfolio["Weight"] > 0.001].sort_values("Weight", ascending=False)

    ret, vol, sharpe = ef.portfolio_performance(verbose=False)
    print("\nOptimization succeeded:")
    print(f"Expected annual return: {ret*100:.2f}%")
    print(f"Annual volatility: {vol*100:.2f}%")
    print(f"Sharpe ratio: {sharpe:.3f}")
    print("\nPortfolio allocations:")
    print(portfolio.to_string(index=False))

    portfolio.to_csv("november_portfolio_max_sharpe.csv", index=False)
    print("\nSaved portfolio to 'november_portfolio_max_sharpe.csv'")

except Exception as e:
    print("Optimization failed:", e)
    print("Falling back to equal-weight allocation among topk.")
    eq_weights = np.ones(len(topk)) / len(topk)
    portfolio = pd.DataFrame({"Stock": topk, "Weight": eq_weights})
    print(portfolio)
    portfolio.to_csv("november_portfolio_fallback.csv", index=False)
    print("Saved fallback portfolio to 'november_portfolio_fallback.csv'")

# Diagnostics (top 10 final scores)
print("\nTop 10 tickers by final_score (diagnostic):")
diag = pd.DataFrame({
    "final_score": final_score,
    "mv": z_mv.reindex(final_score.index),
    "rolling_sharpe": z_rs.reindex(final_score.index),
    "beta_inv": z_beta.reindex(final_score.index),
    "drawdown_inv": z_dd.reindex(final_score.index),
    "dynamic_mom": z_dynmom.reindex(final_score.index),
    "sentiment": z_sent.reindex(final_score.index),
    "mu_long": mu_long.reindex(final_score.index),
    "mu_recent": mu_recent.reindex(final_score.index),
})
print(diag.sort_values("final_score", ascending=False).head(10).round(4))
