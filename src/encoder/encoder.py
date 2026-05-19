"""V-JEPA 2 encoder wrapper for single-frame feature extraction.

Loads facebook/vjepa2-vith-fpc64-256 from Hugging Face Hub and exposes
a simple interface that takes a batch of images (B, C, H, W) and returns
patch embeddings.
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
            (B, N_patches, embed_dim) tensor of embeddings.
        """
        with torch.no_grad():
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)  # (B, 1, C, H, W)
            output = self.model(pixel_values_videos=frames.to(dtype=self.dtype))
        return output.last_hidden_state.squeeze(1)  # (B, N_patches, embed_dim)

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_patches(self) -> int:
        config = self.model.config
        img_size = config.image_size
        patch_size = config.patch_size
        return (img_size // patch_size) ** 2
