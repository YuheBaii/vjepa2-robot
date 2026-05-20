"""V-JEPA 2 encoder wrapper for single-frame feature extraction.

Loads facebook/vjepa2-vith-fpc64-256 from a local checkpoint directory and
exposes a simple interface: input (B, C, H, W) → output (B, 256, 1280).

Dimension note:
  The model uses tubelet_size=2 in the time axis, so total tokens =
  ceil(T / 2) * n_spatial.  With T=1 (single image):
    ceil(1/2) = 1 tubelet → 1 * 256 = 256 tokens → output (B, 256, 1280).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VJEPA2Model


class VJEPA2Encoder(nn.Module):
    """Thin wrapper around V-JEPA 2 ViT-H that extracts frame-level features."""

    def __init__(
        self,
        checkpoint_dir: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.model = VJEPA2Model.from_pretrained(checkpoint_dir, torch_dtype=dtype)
        self.model.eval()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings from a batch of images.

        Args:
            frames: (B, C, H, W) float tensor, pixel values in [0, 1].

        Returns:
            (B, 256, 1280) tensor — 256 spatial patches × 1280 embed_dim.
        """
        with torch.no_grad():
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)  # (B, C, H, W) → (B, 1, C, H, W)
            output = self.model(pixel_values_videos=frames.to(dtype=self.dtype))
        return output.last_hidden_state  # (B, 256, 1280) for T=1

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_patches(self) -> int:
        config = self.model.config
        return (config.image_size // config.patch_size) ** 2
