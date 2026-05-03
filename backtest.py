from typing import Dict, List, Tuple

import numpy as np
import torch

from agents.trader import MultiModalPolicy
from env.portfolio_env import EpisodeStats, PortfolioEnv


def compute_metrics(equity_curve: List[float], daily_returns: List[float]) -> Dict[str, float]:
    """
    Compute standard portfolio performance metrics.

    Args:
        equity_curve: List of portfolio values over time.
        daily_returns: List of daily net returns.

    Returns:
        Dict containing cumulative return, Sharpe ratio, Sortino ratio,
        max drawdown, and Calmar ratio.
    """
    vals = np.asarray(equity_curve, dtype=np.float64)
    dr = np.asarray(daily_returns, dtype=np.float64)

    cumulative_return = float((vals[-1] / vals[0] - 1.0) * 100.0)

    # Sharpe ratio: annualised risk-adjusted return
    sharpe = float(np.sqrt(252.0) * dr.mean() / (dr.std() + 1e-9))

    # Sortino ratio: like Sharpe but only penalises downside volatility
    downside = dr[dr < 0]
    downside_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-9
    sortino = float(np.sqrt(252.0) * dr.mean() / (downside_std + 1e-9))

    # Max drawdown
    peaks = np.maximum.accumulate(vals)
    drawdowns = (vals - peaks) / (peaks + 1e-9)
    max_drawdown = float(drawdowns.min() * 100.0)

    # Calmar ratio: annualised return divided by max drawdown (absolute)
    annualised_return = dr.mean() * 252.0 * 100.0
    calmar = float(annualised_return / (abs(max_drawdown) + 1e-9))

    return {
        "cumulative_return_pct": round(cumulative_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "calmar_ratio": round(calmar, 4),
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
