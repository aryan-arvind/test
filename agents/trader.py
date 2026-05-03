from typing import Dict

import torch
import torch.nn as nn


class PriceLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class FeedForwardEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalPolicy(nn.Module):
    """PPO actor-critic with 3 separate encoders and configurable ablation mode."""

    def __init__(
        self,
        price_dim: int,
        sentiment_dim: int,
        volume_dim: int,
        n_assets: int,
        mode: str = "all",
    ) -> None:
        super().__init__()
        valid_modes = {"price", "sentiment", "volume", "all"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode {mode}; expected one of {sorted(valid_modes)}")

        self.mode = mode
        self.n_assets = n_assets

        self.price_encoder = PriceLSTM(input_dim=price_dim, hidden_dim=64, output_dim=32)
        self.sentiment_encoder = FeedForwardEncoder(input_dim=sentiment_dim, hidden_dim=32, output_dim=32)
        self.volume_encoder = FeedForwardEncoder(input_dim=volume_dim, hidden_dim=32, output_dim=32)

        if mode == "all":
            fusion_dim = 96
        else:
            fusion_dim = 32

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        action_dim = n_assets * 2
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(64, 1)

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        price_z = self.price_encoder(obs["price_seq"])
        sent_z = self.sentiment_encoder(obs["sentiment"])
        vol_z = self.volume_encoder(obs["volume"])

        if self.mode == "price":
            fused_in = price_z
        elif self.mode == "sentiment":
            fused_in = sent_z
        elif self.mode == "volume":
            fused_in = vol_z
        else:
            fused_in = torch.cat([price_z, sent_z, vol_z], dim=-1)

        return self.fusion(fused_in)

    def forward(self, obs: Dict[str, torch.Tensor]):
        latent = self._encode(obs)
        mean = self.actor_mean(latent)
        log_std = torch.clamp(self.actor_log_std, min=-5.0, max=2.0)
        value = self.critic(latent).squeeze(-1)
        return mean, log_std, value

    def get_dist_and_value(self, obs: Dict[str, torch.Tensor]):
        mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist, value

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        dist, value = self.get_dist_and_value(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
