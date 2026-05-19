# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

V-JEPA 2-AC Robot: a robotic manipulation pipeline built on top of Meta's V-JEPA 2 (ViT-Huge, `facebook/vjepa2-vith-fpc64-256`) as a frozen visual encoder. The project implements two complementary pipelines:

1. **Decoder pipeline** (current focus): Frozen JEPA encoder → MLP Decoder → predict end-effector position + gripper contact, trained on RLBench `PickAndLift` demonstrations.
2. **Planner pipeline** (scaffolded): Frozen JEPA encoder → Action-Conditioned Predictor → CEM Planner → robot actions, using robosuite environments.

## Environment Setup

Conda env: `vjepa2-robot`. Activate before running anything:
```bash
conda activate vjepa2-robot
```

Install all dependencies (PyTorch CUDA 12.6, transformers from GitHub, robosuite, h5py, etc.):
```bash
bash scripts/install_deps.sh
```

## Common Commands

### Verify full setup (encoder + robosuite):
```bash
python scripts/verify_setup.py
```

### Collect RLBench demonstrations (requires CoppeliaSim running):
```bash
# CoppeliaSim env vars are set automatically by the wrapper script
bash scripts/run_with_rlbench.sh scripts/collect_rlbench_data.py \
    --num_demos 50 --output data/rlbench_pick_and_lift.h5
```

### Train the MLP Decoder:
```bash
python scripts/train_decoder.py --config configs/decoder.yaml
python scripts/train_decoder.py --config configs/decoder.yaml --no_wandb   # skip W&B
python scripts/train_decoder.py --config configs/decoder.yaml --device cuda:0
```

### Inspect collected data:
```bash
python data/read_h5.py
```

## Architecture

### Encoder (`src/encoder/encoder.py`)
- `VJEPA2Encoder`: wraps `VJEPA2Model` from HuggingFace, loaded from `checkpoints/` in FP16.
- Input: `(B, C, H, W)` float images in `[0, 1]`; auto-adds the temporal dim internally.
- Output: `(B, 256, 1280)` — 256 patches × 1280 embed_dim.
- **Always frozen** — never update its weights.

### Decoder (`src/decoder/mlp_decoder.py`)
- `AttentionPool`: learnable query token attends to the 256 patches → `(B, 1280)`.
- `MLPDecoder`: pool → 3×Linear(→512)+ReLU trunk → two heads:
  - `ee_head`: `Linear(512→3)` — end-effector 3D position (MSE loss, weight 1.0)
  - `contact_head`: `Linear(512→1)` — gripper contact raw logit (BCEWithLogits, weight 0.5)
- ~2M trainable parameters total.

### Predictor (`src/predictor/predictor.py`)
- `ActionConditionedPredictor`: takes `z (B, N, 1280)` + `a (B, 7)`, outputs `z + delta_z` (residual).
- Used in the CEM planning loop — not yet integrated with training data.

### Planner (`src/planner/planner.py`)
- `CEMPlanner`: Cross-Entropy Method over a planning horizon. At each CEM iter, samples `num_samples` action sequences, rolls out through the predictor, scores with a cost function, refits from top `num_elites`.

### Data
- `src/data/rlbench_dataset.py` — `RLBenchDataset`: loads HDF5 with keys `front_rgb (N,256,256,3)`, `ee_pos (N,3)`, `contact (N,)`. Lazy-opens the file per worker (DataLoader-safe). Contact = `1.0 - gripper_open`.
- `src/data/dataset.py` — `TrajectoryDataset`: loads sequence chunks `(images, actions)` for the predictor pipeline (HDF5 keys `images`, `actions`).

### Environment
- `src/env/env_wrapper.py` — `RobosuiteEnv`: thin wrapper around robosuite for the Lift task (Franka Panda). Returns `(H, W, 3)` uint8 RGB from `frontview` camera.

## Key Configuration (`configs/`)

- `configs/default.yaml` — full pipeline defaults (encoder, predictor, planner, robosuite env, training).
- `configs/decoder.yaml` — decoder-specific training: encoder FP16 from `checkpoints/`, AdamW lr=1e-3, CosineAnnealingLR, 50 epochs, 80/20 train/val split.

## Checkpoints

- JEPA encoder: stored in `checkpoints/` (loaded as a local HuggingFace model — place `config.json`, `model.safetensors`, etc. there).
- Decoder checkpoints saved to `checkpoints/decoder/best_decoder.pt` and `final_decoder.pt`.

## RLBench / CoppeliaSim Notes

- CoppeliaSim must be installed at `COPPELIASIM_ROOT` (default: `/home/yuhe/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`).
- Use `scripts/run_with_rlbench.sh <script>` to set the required env vars (`COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, `QT_PLUGIN_PATH`, `DISPLAY`).
- `collect_rlbench_data.py` uses `live_demos=True` + a `_step_callback` to record every step; failed variations are skipped and collection continues.
- Data is saved as gzip-compressed HDF5.

## W&B Integration

Training logs to project `vjepa2-robot-decoder` by default. Pass `--no_wandb` to disable.
