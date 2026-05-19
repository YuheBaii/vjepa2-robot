#!/usr/bin/env python
"""Verify the V-JEPA 2-AC robot setup end-to-end.

Checks:
  1. Load V-JEPA 2 ViT-H encoder (FP16), print param count & GPU memory.
  2. Create random 256x256 image tensor x16 frames, encode, print output shape & peak memory.
  3. Initialize robosuite Lift env (Franka Panda), reset, render 256x256 RGB, save to results/.
  4. Encode the rendered frame, print output shape.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _gpu_mem_mb(device: torch.device) -> float:
    """Return allocated GPU memory in MiB."""
    return torch.cuda.memory_allocated(device) / (1024**2)


def _peak_mem_mb(device: torch.device) -> float:
    """Return peak GPU memory in MiB since last reset."""
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _reset_mem(device: torch.device):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Precision: {dtype}")
    print()

    # ── Step 1: Load encoder ───────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading V-JEPA 2 ViT-H encoder (FP16)")
    print("=" * 60)

    _reset_mem(device)

    from src.encoder import VJEPA2Encoder

    t0 = time.time()
    encoder = VJEPA2Encoder(CHECKPOINT_DIR, dtype=dtype)
    encoder.to(device)
    print(f"  Load time: {time.time() - t0:.1f}s")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    print(f"  GPU memory (params): {_gpu_mem_mb(device):.1f} MiB")

    # ── Step 2: Random frames ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 2: Encoding 16 random 256x256 frames")
    print("=" * 60)

    _reset_mem(device)
    B, T, C, H, W = 1, 16, 3, 256, 256
    fake_frames = torch.rand(B, C, H, W, device=device, dtype=torch.float32)

    # Encode each frame one by one (we process single frames, not video).
    all_embeds = []
    t0 = time.time()
    for i in range(T):
        z = encoder(fake_frames)  # (1, N_patches, embed_dim)
        all_embeds.append(z)
        if i == 0:
            print(f"  Frame 0 output shape: {tuple(z.shape)}")
    elapsed = time.time() - t0
    print(f"  Encoded {T} frames in {elapsed:.2f}s ({elapsed / T * 1000:.1f} ms/frame)")
    print(f"  Peak GPU memory (inference): {_peak_mem_mb(device):.1f} MiB")

    # ── Step 3: robosuite env ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 3: robosuite Lift environment (Franka Panda)")
    print("=" * 60)

    from src.env import RobosuiteEnv

    env = RobosuiteEnv(env_name="Lift", render_size=256)
    print(f"  Environment: Lift | Robot: Panda | Render: 256x256")

    img = env.reset()
    print(f"  Render shape: {img.shape}  dtype: {img.dtype}  range: [{img.min()}, {img.max()}]")

    from PIL import Image
    save_path = os.path.join(RESULTS_DIR, "test_render.png")
    Image.fromarray(img).save(save_path)
    print(f"  Saved to: {save_path}")

    env.close()

    # ── Step 4: Encode rendered frame ──────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 4: Encoding the rendered RGB frame")
    print("=" * 60)

    # Convert uint8 [0,255] to float32 [0,1], HWC -> CHW, add batch dim
    frame_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    frame_t = frame_t.unsqueeze(0).to(device)  # (1, 3, 256, 256)

    _reset_mem(device)
    with torch.no_grad():
        z_real = encoder(frame_t)
    print(f"  Output shape: {tuple(z_real.shape)}")
    print(f"  Embed dim: {z_real.shape[-1]}  Num patches: {z_real.shape[1]}")
    print(f"  Peak GPU memory: {_peak_mem_mb(device):.1f} MiB")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("VERIFICATION PASSED")
    print("=" * 60)
    print(f"  Encoder: V-JEPA 2 ViT-H  {n_params / 1e6:.0f}M params")
    print(f"  env: robosuite Lift (Panda)")
    print(f"  Pipeline: render -> encode -> features  OK")


if __name__ == "__main__":
    main()
