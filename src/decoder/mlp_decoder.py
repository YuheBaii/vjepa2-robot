"""MLP Decoder that predicts robot state from JEPA encoder features.

Takes patch-level embeddings z_t (B, N_patches, embed_dim) from the frozen
V-JEPA 2 encoder and predicts end-effector position and gripper contact state.
Uses attention pooling to focus on task-relevant image regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Learnable attention pooling over patch embeddings.

    Instead of mean pooling (equal weight per patch), this learns a query
    token that attends to the most relevant patches for the downstream tasks.
    """

    def __init__(self, embed_dim: int = 1280) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool patch embeddings via attention.

        Args:
            x: (B, N_patches, embed_dim) patch embeddings.

        Returns:
            (B, embed_dim) pooled feature vector.
        """
        B = x.shape[0]
        # x: (B, N, D), query: (1, 1, D), key: (B, N, D)
        q = self.query.expand(B, -1, -1)           # (B, 1, D)
        k = self.key(x)                            # (B, N, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, N)
        attn = attn.softmax(dim=-1)                     # (B, 1, N)
        z = attn @ x                                    # (B, 1, D)
        return z.squeeze(1)                             # (B, D)


class MLPDecoder(nn.Module):
    """MLP decoder with attention pooling and two task-specific heads.

    Input:  z_t (B, N_patches, 1280) from frozen JEPA encoder
    Output: ee_pos (B, 3), contact_logit (B, 1)
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.pool = AttentionPool(embed_dim)

        trunk_layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(in_dim, hidden_dim))
            trunk_layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        self.ee_head = nn.Linear(hidden_dim, 3)
        self.contact_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, z_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            z_t: (B, N_patches, embed_dim) patch embeddings from encoder.

        Returns:
            ee_pos:        (B, 3) predicted end-effector 3D position.
            contact_logit: (B, 1) raw logit for gripper contact.
        """
        z = self.pool(z_t)                  # (B, embed_dim)  attention pooling
        h = self.trunk(z)                   # (B, hidden_dim)
        ee_pos = self.ee_head(h)            # (B, 3)
        contact_logit = self.contact_head(h)  # (B, 1)
        return ee_pos, contact_logit
