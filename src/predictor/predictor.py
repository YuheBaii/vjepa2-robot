"""Action-Conditioned Predictor (AC-Predictor) for V-JEPA 2.

Takes current state embeddings + action latent and predicts future state embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedPredictor(nn.Module):
    """Predicts future latent state given current latent + action."""

    def __init__(
        self,
        embed_dim: int = 1280,
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim

        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.predictor = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict next latent state.

        Args:
            z: (B, N_patches, embed_dim) current state embeddings.
            a: (B, action_dim) action vectors.

        Returns:
            (B, N_patches, embed_dim) predicted next-state embeddings.
        """
        a_h = self.action_proj(a)                     # (B, hidden_dim)
        a_h = a_h.unsqueeze(1).expand(-1, z.size(1), -1)  # (B, N_patches, hidden_dim)
        fused = self.fusion(torch.cat([z, a_h], dim=-1))  # (B, N_patches, hidden_dim)
        delta_z = self.predictor(fused)                # (B, N_patches, embed_dim)
        return z + delta_z
