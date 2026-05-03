# Faculty Presentation Notes (DL + RL MVP)

## 1) What this project is
A multimodal Deep Learning + Reinforcement Learning trading MVP that:
- uses Indian equities (RELIANCE, TCS, HDFCBANK, INFY)
- builds technical, sentiment, and volume feature streams
- trains a custom PPO policy in PyTorch (from scratch)
- evaluates via backtesting and ablation studies
- serves metrics and inferred portfolio weights to a React dashboard via FastAPI

## 2) Data pipeline
File: `data/load_data.py`
- Date window: 2021-01-01 to 2025-01-01
- Symbols: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS
- Features:
  - Price/technical: returns, SMA ratios, RSI, MACD
  - Volume: z-score and pct change
  - Sentiment: monthly Google News RSS headline sentiment mapped to daily
- Splits:
  - Train <= 2022-12-31
  - Backtest >= 2023-01-01
- Reliability:
  - local cache for market features
  - synthetic fallback if market download fails (offline demo safe)

## 3) Model architecture (DL)
File: `agents/trader.py`
- Price encoder: LSTM
- Sentiment encoder: feed-forward network
- Volume encoder: feed-forward network
- Fusion + actor-critic heads:
  - Actor outputs buy/sell/hold signal + position size per asset
  - Critic outputs state value
- Ablation modes:
  - price only
  - sentiment only
  - volume only
  - all modalities

## 4) RL environment and reward
File: `env/portfolio_env.py`
- Portfolio accounting with transaction costs
- Action decoding into target positions
- Reward uses Differential Sharpe Ratio (risk-adjusted objective)
- Tracks equity curve, daily returns, weight history

## 5) PPO training and ablation
File: `train.py`
- Custom PPO loop with:
  - rollouts
  - GAE advantages
  - clipping objective
  - entropy regularization
  - gradient clipping
- Trains all 4 ablation modes and picks best by Sharpe
- Writes artifacts:
  - `artifacts/ppo_multimodal.pt`
  - `artifacts/ablation_results.json`
  - `artifacts/latest_metrics.json`

## 6) Backtest metrics
File: `backtest.py`
- cumulative return (%)
- annualized Sharpe ratio
- maximum drawdown (%)

## 7) API + frontend integration
File: `backend_api.py`
- Endpoint: `GET /api/dashboard`
- Returns:
  - current portfolio cards (price, change, inferred weight)
  - market stats (NIFTY, VIX)
  - metrics from backtest
  - chart points
- Backend now caches payload to improve response speed for live demo

File: `frontend/src/App.jsx`
- Dashboard fetches `/api/dashboard`
- Displays real backend metrics and portfolio weights (not placeholders)

## 8) Current run outcome (generated now)
From latest training/ablation:
- Best mode: `price`
- Cumulative return: `+2.79%`
- Sharpe: `0.99`
- Max drawdown: `-1.07%`

## 9) How to explain contribution clearly
1. "I replaced prototype placeholders with a full custom DL+RL stack in PyTorch PPO."
2. "I fused three modalities (price, sentiment, volume) and validated each via ablation."
3. "I used Differential Sharpe reward so learning optimizes risk-adjusted returns, not just raw profit."
4. "I integrated backend metrics and policy-inferred weights into a live frontend endpoint for end-to-end reproducibility."

## 10) Known MVP limitations (say this confidently)
- Sentiment model is lexicon-based headline scoring (simple but transparent)
- Universe size is 4 equities (kept small for reproducible MVP)
- Backtest still uses simplified execution assumptions (no slippage model)

These are expected MVP constraints and clear extension points for future work.
