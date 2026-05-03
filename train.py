import json
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from agents.trader import MultiModalPolicy
from backtest import backtest_model
from data.load_data import DEFAULT_SYMBOLS, LOOKBACK, load_market_dataset
from env.portfolio_env import PortfolioEnv


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "ppo_multimodal.pt"
ABLATION_PATH = ARTIFACT_DIR / "ablation_results.json"
METRICS_PATH = ARTIFACT_DIR / "latest_metrics.json"


@dataclass
class RolloutBatch:
    obs_price: torch.Tensor
    obs_sentiment: torch.Tensor
    obs_volume: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def _to_torch_obs(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "price_seq": torch.tensor(obs["price_seq"], dtype=torch.float32, device=device).unsqueeze(0),
        "sentiment": torch.tensor(obs["sentiment"], dtype=torch.float32, device=device).unsqueeze(0),
        "volume": torch.tensor(obs["volume"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def _stack_obs(observations: List[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    price = np.stack([o["price_seq"] for o in observations], axis=0)
    sentiment = np.stack([o["sentiment"] for o in observations], axis=0)
    volume = np.stack([o["volume"] for o in observations], axis=0)
    return {
        "price_seq": torch.tensor(price, dtype=torch.float32, device=device),
        "sentiment": torch.tensor(sentiment, dtype=torch.float32, device=device),
        "volume": torch.tensor(volume, dtype=torch.float32, device=device),
    }


def _collect_rollout(
    env: PortfolioEnv,
    policy: MultiModalPolicy,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> RolloutBatch:
    obs_buffer: List[Dict[str, np.ndarray]] = []
    action_buffer: List[np.ndarray] = []
    log_prob_buffer: List[float] = []
    reward_buffer: List[float] = []
    value_buffer: List[float] = []

    obs = env.reset()
    done = False
    while not done:
        torch_obs = _to_torch_obs(obs, device=device)
        action_t, log_prob_t, value_t = policy.act(torch_obs, deterministic=False)
        action = action_t.squeeze(0).cpu().numpy()

        next_obs, reward, done, _ = env.step(action)

        obs_buffer.append(obs)
        action_buffer.append(action)
        log_prob_buffer.append(float(log_prob_t.item()))
        reward_buffer.append(float(reward))
        value_buffer.append(float(value_t.item()))

        obs = next_obs

    values = np.asarray(value_buffer, dtype=np.float32)
    rewards = np.asarray(reward_buffer, dtype=np.float32)
    advantages = np.zeros_like(rewards)

    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t = _stack_obs(obs_buffer, device=device)
    return RolloutBatch(
        obs_price=obs_t["price_seq"],
        obs_sentiment=obs_t["sentiment"],
        obs_volume=obs_t["volume"],
        actions=torch.tensor(np.asarray(action_buffer), dtype=torch.float32, device=device),
        old_log_probs=torch.tensor(np.asarray(log_prob_buffer), dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
    )


def _train_ppo(
    policy: MultiModalPolicy,
    env: PortfolioEnv,
    epochs: int = 12,
    ppo_epochs: int = 8,
    mini_batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lr: float = 3e-4,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for _ in range(epochs):
        batch = _collect_rollout(env, policy, device=device, gamma=gamma, gae_lambda=gae_lambda)

        data_size = batch.actions.shape[0]
        idx = torch.randperm(data_size, device=device)

        for _ in range(ppo_epochs):
            for start in range(0, data_size, mini_batch_size):
                end = min(start + mini_batch_size, data_size)
                mb = idx[start:end]

                obs_mb = {
                    "price_seq": batch.obs_price[mb],
                    "sentiment": batch.obs_sentiment[mb],
                    "volume": batch.obs_volume[mb],
                }
                action_mb = batch.actions[mb]
                old_log_prob_mb = batch.old_log_probs[mb]
                return_mb = batch.returns[mb]
                advantage_mb = batch.advantages[mb]

                dist, value = policy.get_dist_and_value(obs_mb)
                log_prob = dist.log_prob(action_mb).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(log_prob - old_log_prob_mb)
                surr1 = ratio * advantage_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, return_mb)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()


def _build_policy_from_dataset(mode: str, dataset) -> MultiModalPolicy:
    sample = dataset.train_obs[0]
    price_dim = sample["price_seq"].shape[1]
    sentiment_dim = sample["sentiment"].shape[0]
    volume_dim = sample["volume"].shape[0]
    n_assets = dataset.train_next_returns.shape[1]
    return MultiModalPolicy(
        price_dim=price_dim,
        sentiment_dim=sentiment_dim,
        volume_dim=volume_dim,
        n_assets=n_assets,
        mode=mode,
    )


def train_and_run_ablation(
    symbols: List[str] = None,
    lookback: int = LOOKBACK,
    epochs: int = 10,
) -> Dict[str, Dict[str, float]]:
    symbols = symbols or DEFAULT_SYMBOLS
    dataset = load_market_dataset(symbols=symbols, lookback=lookback)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    ablation_results: Dict[str, Dict[str, float]] = {}
    best_mode = None
    best_sharpe = -1e9
    best_state = None

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for mode in ["price", "sentiment", "volume", "all"]:
        print(f"Training mode: {mode}")
        policy = _build_policy_from_dataset(mode=mode, dataset=dataset)
        train_env = PortfolioEnv(dataset.train_obs, dataset.train_next_returns)
        
        # Train with eval splitting
        for epoch in range(1, epochs + 1):
            _train_ppo(policy=policy, env=train_env, epochs=1)
            
            # Evaluate on the test split
            eval_metrics, _ = backtest_model(policy, dataset.test_obs, dataset.test_next_returns)
            
            # Save eval metrics to the logs folder
            log_file = f"logs/eval_{mode}_{timestamp}.jsonl"
            with open(log_file, "a") as f:
                log_entry = {"epoch": epoch, "sharpe_ratio": eval_metrics["sharpe_ratio"], "return": eval_metrics["cumulative_return"]}
                f.write(json.dumps(log_entry) + "\n")

        metrics, _stats = backtest_model(policy, dataset.test_obs, dataset.test_next_returns)
        ablation_results[mode] = metrics

        if metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_mode = mode
            best_state = policy.state_dict()

    if best_state is None or best_mode is None:
        raise RuntimeError("Ablation did not produce a valid model")

    torch.save(
        {
            "mode": best_mode,
            "state_dict": best_state,
            "symbols": symbols,
            "lookback": lookback,
            "price_dim": dataset.train_obs[0]["price_seq"].shape[1],
            "sentiment_dim": dataset.train_obs[0]["sentiment"].shape[0],
            "volume_dim": dataset.train_obs[0]["volume"].shape[0],
            "n_assets": dataset.train_next_returns.shape[1],
        },
        MODEL_PATH,
    )

    ABLATION_PATH.write_text(json.dumps(ablation_results, indent=2), encoding="utf-8")

    best_metrics = dict(ablation_results[best_mode])
    best_metrics["mode"] = best_mode
    METRICS_PATH.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    return {
        "ablation": ablation_results,
        "best": best_metrics,
    }


if __name__ == "__main__":
    output = train_and_run_ablation(epochs=4)
    print(json.dumps(output, indent=2))
