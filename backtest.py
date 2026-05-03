from typing import Dict, List, Tuple

import numpy as np
import torch

from agents.trader import MultiModalPolicy
from env.portfolio_env import EpisodeStats, PortfolioEnv


def compute_metrics(equity_curve: List[float], daily_returns: List[float]) -> Dict[str, float]:
    vals = np.asarray(equity_curve, dtype=np.float64)
    dr = np.asarray(daily_returns, dtype=np.float64)

    cumulative_return = float((vals[-1] / vals[0] - 1.0) * 100.0)
    sharpe = float(np.sqrt(252.0) * dr.mean() / (dr.std() + 1e-9))
    peaks = np.maximum.accumulate(vals)
    drawdowns = (vals - peaks) / (peaks + 1e-9)
    max_drawdown = float(drawdowns.min() * 100.0)

    return {
        "cumulative_return_pct": cumulative_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown,
    }


@torch.no_grad()
def backtest_model(
    policy: MultiModalPolicy,
    observations: List[Dict[str, np.ndarray]],
    next_returns: np.ndarray,
) -> Tuple[Dict[str, float], EpisodeStats]:
    device = next(policy.parameters()).device
    env = PortfolioEnv(observations, next_returns)
    obs = env.reset()
    done = False

    while not done:
        torch_obs = {
            "price_seq": torch.tensor(obs["price_seq"], dtype=torch.float32, device=device).unsqueeze(0),
            "sentiment": torch.tensor(obs["sentiment"], dtype=torch.float32, device=device).unsqueeze(0),
            "volume": torch.tensor(obs["volume"], dtype=torch.float32, device=device).unsqueeze(0),
        }
        action, _log_prob, _value = policy.act(torch_obs, deterministic=True)
        obs, _reward, done, _info = env.step(action.squeeze(0).cpu().numpy())

    stats = env.get_episode_stats()
    metrics = compute_metrics(stats.equity_curve, stats.daily_returns)
    return metrics, stats
