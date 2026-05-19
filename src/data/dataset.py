"""Trajectory dataset for stored (image, action) pairs."""

from __future__ import annotations

import h5py
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset that loads trajectory chunks from an HDF5 file."""

    def __init__(self, h5_path: str, seq_len: int = 16):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self._file: h5py.File | None = None
        with h5py.File(h5_path, "r") as f:
            self._len = len(f["images"]) - seq_len

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return max(0, self._len)

    def __getitem__(self, idx: int):
        self._open()
        images = self._file["images"][idx : idx + self.seq_len]   # (T, H, W, 3)
        actions = self._file["actions"][idx : idx + self.seq_len] # (T, action_dim)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        actions = torch.from_numpy(actions).float()
        return images, actions

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
