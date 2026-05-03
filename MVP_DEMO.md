# DL + RL MVP Demo Guide

This project now has a from-scratch PyTorch PPO trading pipeline with:
- real NSE data pull (2021-2024/25 window) for RELIANCE, TCS, HDFCBANK, INFY
- technical + sentiment + volume feature pipeline
- 3-branch model (price LSTM, sentiment FFN, volume FFN)
- PPO actor-critic training with GAE + clipping
- ablation modes: price / sentiment / volume / all
- backtest metrics: cumulative return, Sharpe, max drawdown
- FastAPI endpoint wired to frontend dashboard values

## 1) Use a Python version supported by PyTorch
Recommended: Python 3.10 or 3.11.

If your current environment is 3.14, create a fresh venv with 3.11 and install requirements there.

## 2) Install dependencies
From the `trading_agents` folder:

```powershell
pip install -r requirements.txt
```

If torch install fails, install CPU wheels explicitly:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 3) Train + ablation + artifact generation

```powershell
python run.py --mode train --epochs 4
```

Artifacts generated:
- `artifacts/ppo_multimodal.pt`
- `artifacts/ablation_results.json`
- `artifacts/latest_metrics.json`

## 4) Start backend API

```powershell
python run.py --mode serve --port 8000
```

Endpoint:
- `GET /api/dashboard`

## 5) Start frontend
In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The dashboard will read live values from backend `metrics` and `portfolio` weights.

## Faculty demo script (2-3 min)
1. Show `artifacts/ablation_results.json` with 4 ablation modes.
2. Start backend (`run.py --mode serve`).
3. Open frontend and refresh once.
4. Explain that displayed return / Sharpe / drawdown come from backtest outputs and weights are inferred from trained policy action head.

## Reliability notes for presentation
- API now caches trained model + backtest payload for faster repeated requests.
- Data loader has an offline synthetic fallback if market API temporarily fails.
- Frontend has chart fallback data so UI never blanks.
