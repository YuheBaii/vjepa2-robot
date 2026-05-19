# vjepa2-robot

Robot manipulation pipeline built on [V-JEPA 2](https://github.com/facebookresearch/vjepa2) (Meta FAIR), using a frozen ViT-Huge encoder to extract visual features for downstream robot learning tasks.

## Overview

Two complementary pipelines are implemented:

**Decoder pipeline** — supervised imitation learning on RLBench demonstrations:
```
RLBench PickAndLift  →  HDF5 dataset  →  Frozen JEPA Encoder  →  MLP Decoder  →  ee_pos + contact
```

**Planner pipeline** — model-predictive control via learned dynamics (scaffolded):
```
robosuite observation  →  Frozen JEPA Encoder  →  AC-Predictor  →  CEM Planner  →  action
```

## Requirements

- CUDA-capable GPU (≥8 GB VRAM recommended for ViT-H in FP16)
- [CoppeliaSim Edu V4.1](https://www.coppeliarobotics.com/) — required for RLBench data collection
- Conda

## Installation

```bash
conda create -n vjepa2-robot python=3.10
conda activate vjepa2-robot
bash scripts/install_deps.sh
```

Dependencies installed: PyTorch (CUDA 12.6), transformers (from GitHub), robosuite, mujoco, h5py, wandb, tqdm.

## Encoder Checkpoint

Download the V-JEPA 2 ViT-H model from HuggingFace and place the files under `checkpoints/`:

```bash
# Using huggingface-cli
huggingface-cli download facebook/vjepa2-vith-fpc64-256 --local-dir checkpoints
```

The encoder is always **frozen** during training — only the decoder or predictor weights are updated.

## Usage

### 1. Verify setup

```bash
python scripts/verify_setup.py
```

Checks encoder loading, GPU memory, and robosuite environment.

### 2. Collect RLBench demonstrations

Requires CoppeliaSim running. The wrapper script sets the necessary environment variables:

```bash
bash scripts/run_with_rlbench.sh scripts/collect_rlbench_data.py \
    --num_demos 50 \
    --output data/rlbench_pick_and_lift.h5
```

Each demo records `front_rgb (N, 256, 256, 3)`, `ee_pos (N, 3)`, and `contact (N,)` at every timestep.

### 3. Train the MLP Decoder

```bash
python scripts/train_decoder.py --config configs/decoder.yaml

# Options
python scripts/train_decoder.py --config configs/decoder.yaml --no_wandb   # disable W&B logging
python scripts/train_decoder.py --config configs/decoder.yaml --device cuda:1
```

Training logs to W&B project `vjepa2-robot-decoder` by default. Checkpoints are saved to `checkpoints/decoder/`:
- `best_decoder.pt` — lowest validation loss
- `final_decoder.pt` — last epoch

## Architecture

### VJEPA2Encoder
Wraps `facebook/vjepa2-vith-fpc64-256` (ViT-Huge, ~630M params). Input: `(B, 3, 256, 256)` → Output: `(B, 256, 1280)` patch embeddings. Runs in FP16.

### MLPDecoder
- **AttentionPool**: learnable query token attends over 256 patch embeddings → `(B, 1280)`
- **Trunk**: 3 × Linear(→512) + ReLU
- **Heads**: `ee_head` → `(B, 3)` position; `contact_head` → `(B, 1)` logit
- ~2M trainable parameters
- Loss: `MSE(ee_pos) × 1.0 + BCEWithLogits(contact) × 0.5`

### ActionConditionedPredictor
Takes `z (B, N, 1280)` + `a (B, 7)` → predicts `z_next` as a residual. Used in the CEM planning loop.

### CEMPlanner
Cross-Entropy Method over a configurable horizon. Iteratively refits a Gaussian over action sequences using the top-k elite rollouts scored by a user-provided cost function.

## Configuration

| File | Purpose |
|------|---------|
| `configs/decoder.yaml` | Decoder training (lr, epochs, loss weights, data path) |
| `configs/default.yaml` | Full pipeline defaults (encoder, predictor, planner, robosuite env) |

## Citation

```bibtex
@techreport{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and others},
  institution={FAIR at Meta},
  year={2025}
}
```
