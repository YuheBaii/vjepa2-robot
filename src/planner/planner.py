"""CEM (Cross-Entropy Method) planner for model-predictive control."""

from __future__ import annotations

import torch
import numpy as np


class CEMPlanner:
    """CEM-based planner that optimizes action sequences using a learned dynamics model."""

    def __init__(
        self,
        action_dim: int = 7,
        horizon: int = 10,
        num_samples: int = 200,
        num_elites: int = 20,
        num_iters: int = 3,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        device: str = "cuda",
    ) -> None:
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.device = device

        self.action_low = (
            torch.tensor(action_low, dtype=torch.float32, device=device)
            if action_low is not None
            else torch.full((action_dim,), -1.0, device=device)
        )
        self.action_high = (
            torch.tensor(action_high, dtype=torch.float32, device=device)
            if action_high is not None
            else torch.full((action_dim,), 1.0, device=device)
        )

    def plan(
        self,
        z0: torch.Tensor,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        cost_fn,
    ) -> torch.Tensor:
        """Return the best first action found by CEM.

        Args:
            z0: (1, N_patches, embed_dim) current state embedding.
            encoder: V-JEPA 2 encoder (unused directly; kept for interface compatibility).
            predictor: ActionConditionedPredictor.
            cost_fn: callable (z_goal, z_pred) -> scalar cost.

        Returns:
            (action_dim,) best first action.
        """
        B = z0.size(0)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        for _ in range(self.num_iters):
            samples = mean + std * torch.randn(
                self.num_samples, self.horizon, self.action_dim, device=self.device
            )
            samples = torch.clamp(samples, self.action_low, self.action_high)

            costs = []
            z = z0.expand(self.num_samples, -1, -1)
            for t in range(self.horizon):
                a = samples[:, t]  # (num_samples, action_dim)
                z = predictor(z, a)
                costs.append(cost_fn(z))

            costs = torch.stack(costs, dim=-1).sum(dim=-1)  # (num_samples,)
            _, elite_idx = torch.topk(costs, self.num_elites, largest=False)
            elites = samples[elite_idx]  # (num_elites, horizon, action_dim)
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=1e-6)

        return mean[0]  # first action of the refined mean
