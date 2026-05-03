from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EpisodeStats:
    equity_curve: List[float]
    daily_returns: List[float]
    weights_history: List[np.ndarray]


class PortfolioEnv:
    """Custom portfolio environment with Differential Sharpe Ratio reward."""

    def __init__(
        self,
        observations: List[Dict[str, np.ndarray]],
        next_returns: np.ndarray,
        max_position: float = 1.0,
        transaction_cost: float = 0.001,
        dsr_eta: float = 0.02,
    ) -> None:
        self.observations = observations
        self.next_returns = next_returns.astype(np.float32)
        self.n_assets = self.next_returns.shape[1]
        self.max_position = float(max_position)
        self.transaction_cost = float(transaction_cost)
        self.dsr_eta = float(dsr_eta)
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.equity = 1.0
        self.a = 0.0
        self.b = 1e-6
        self.equity_curve = [self.equity]
        self.daily_returns = []
        self.weights_history = [self.positions.copy()]
        return self.observations[self.t]

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """
        Decodes the continuous action vector into target portfolio weights.
        
        Args:
            action (np.ndarray): The raw action vector output by the policy.
            
        Returns:
            np.ndarray: The target position weights for each asset.
        """
        signal = action[: self.n_assets]
        size = 1.0 / (1.0 + np.exp(-action[self.n_assets :]))

        decision = np.zeros_like(signal)
        decision[signal > 0.2] = 1.0
        decision[signal < -0.2] = -1.0

        target = decision * size * self.max_position
        return target.astype(np.float32)

    def _differential_sharpe(self, r_t: float) -> float:
        """
        Computes the differential Sharpe ratio reward for the current timestep.
        
        Args:
            r_t (float): The realized net return for the current timestep.
            
        Returns:
            float: The differential Sharpe ratio, serving as the RL reward signal.
        """
        a_prev = self.a
        b_prev = self.b
        eta = self.dsr_eta

        self.a = a_prev + eta * (r_t - a_prev)
        self.b = b_prev + eta * (r_t * r_t - b_prev)

        var = max(b_prev - a_prev * a_prev, 1e-8)
        denom = np.power(var, 1.5) + 1e-8
        numerator = b_prev * (r_t - a_prev) - 0.5 * a_prev * (r_t * r_t - b_prev)
        return float(numerator / denom)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, np.ndarray]]:
        target_positions = self._decode_action(action)
        trade_size = np.abs(target_positions - self.positions)
        costs = self.transaction_cost * float(np.mean(trade_size))

        realized_returns = self.next_returns[self.t]
        pnl = float(np.mean(target_positions * realized_returns))
        net_return = pnl - costs
        self.equity *= 1.0 + net_return

        reward = self._differential_sharpe(net_return)

        self.positions = target_positions
        self.daily_returns.append(net_return)
        self.equity_curve.append(self.equity)
        self.weights_history.append(self.positions.copy())

        self.t += 1
        done = self.t >= len(self.observations)
        next_obs = self.observations[min(self.t, len(self.observations) - 1)]

        info = {
            "net_return": np.array(net_return, dtype=np.float32),
            "positions": self.positions.copy(),
            "equity": np.array(self.equity, dtype=np.float32),
        }
        return next_obs, reward, done, info

    def get_episode_stats(self) -> EpisodeStats:
        return EpisodeStats(
            equity_curve=self.equity_curve,
            daily_returns=self.daily_returns,
            weights_history=self.weights_history,
        )
