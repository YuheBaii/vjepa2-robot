"""RLBench pick_and_lift dataset for supervised decoder training.

Loads pre-collected HDF5 files containing front_rgb images, end-effector
positions, and gripper contact states.
"""

from __future__ import annotations

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RLBenchDataset(Dataset):
    """Dataset that loads RLBench pick_and_lift frames from an HDF5 file.

    Each item returns:
        image:   (3, H, W) float32 tensor, pixel values in [0, 1]
        ee_pos:  (3,) float32 tensor, world-frame end-effector 3D position
        contact: float32 scalar, 0.0 = gripper open, 1.0 = gripper closed
    """

    def __init__(self, h5_path: str, image_size: int = 256) -> None:
        self.h5_path = h5_path
        self.image_size = image_size
        self._file: h5py.File | None = None

        with h5py.File(h5_path, "r") as f:
            self._len = len(f["front_rgb"])

    def _open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self._len

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._open()
        assert self._file is not None

        img = self._file["front_rgb"][idx]          # (H, W, 3) uint8
        ee_pos = self._file["ee_pos"][idx]          # (3,) float32
        contact = self._file["contact"][idx]        # scalar float32

        image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        # Resize to encoder input size if needed
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        ee_pos = torch.from_numpy(ee_pos).float()
        contact = torch.tensor(float(contact), dtype=torch.float32)

        return image, ee_pos, contact

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
