#!/bin/bash
set -euo pipefail

LOG="/home/yuhe/Downloads/vjepa2-robot/install.log"

# Source conda
source /home/yuhe/miniforge3/etc/profile.d/conda.sh
conda activate vjepa2-robot

echo "=== Starting installation at $(date) ===" | tee -a "$LOG"

# PyTorch with CUDA
echo "[1/7] Installing PyTorch..." | tee -a "$LOG"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 2>&1 | tee -a "$LOG"
echo "PyTorch done: $(python -c 'import torch; print(torch.__version__)')" | tee -a "$LOG"

# transformers from GitHub
echo "[2/7] Installing transformers from GitHub..." | tee -a "$LOG"
pip install -U git+https://github.com/huggingface/transformers 2>&1 | tee -a "$LOG"

# Core packages
echo "[3/7] Installing timm, einops..." | tee -a "$LOG"
pip install timm einops 2>&1 | tee -a "$LOG"

# robosuite and mujoco
echo "[4/7] Installing robosuite, mujoco..." | tee -a "$LOG"
pip install robosuite mujoco 2>&1 | tee -a "$LOG"

# h5py
echo "[5/7] Installing h5py..." | tee -a "$LOG"
pip install h5py 2>&1 | tee -a "$LOG"

# scipy
echo "[6/7] Installing scipy..." | tee -a "$LOG"
pip install scipy 2>&1 | tee -a "$LOG"

# visualization and tools
echo "[7/7] Installing matplotlib, tensorboard, jupyter, torchcodec..." | tee -a "$LOG"
pip install matplotlib tensorboard jupyter torchcodec 2>&1 | tee -a "$LOG"

echo "=== All packages installed at $(date) ===" | tee -a "$LOG"
