"""
V-JEPA 2 模型使用指南

涵盖：
  1. 如何加载预训练模型
  2. 输入 / 输出维度
  3. 图像 vs 视频的输入区别
"""

import torch
from transformers import VJEPA2Model, AutoVideoProcessor

CHECKPOINT_DIR = "checkpoints"   # 本地目录，含 config.json + model.safetensors
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# 1. 加载模型
# ─────────────────────────────────────────────
# from_pretrained 接受 HuggingFace Hub ID 或本地目录路径
model = VJEPA2Model.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float16)
model.to(DEVICE).eval()

# 模型基本信息
cfg = model.config
print(f"模型:      ViT-H  hidden_size={cfg.hidden_size}")          # 1280
print(f"patch_size={cfg.patch_size}  image_size={cfg.image_size}") # 16, 256
n_patches = (cfg.image_size // cfg.patch_size) ** 2
print(f"空间 patch 数: {n_patches}")   # (256//16)^2 = 256

# ─────────────────────────────────────────────
# 2. 输入格式
# ─────────────────────────────────────────────
# VJEPA2Model 只接受一个关键字参数：
#   pixel_values_videos: (B, T, C, H, W)  float16/32，值域 [0, 1]
#
# T = 帧数（时间维度），模型内部会在时间轴上做 patch 化再 flatten。
# 对于 ViT-H / fpc64：T 通常为 16 或 64，但任意 T 均可输入。

B, T, C, H, W = 2, 16, 3, 256, 256
video = torch.rand(B, T, C, H, W, dtype=torch.float16, device=DEVICE)

with torch.no_grad():
    out = model(pixel_values_videos=video)

# ─────────────────────────────────────────────
# 3. 输出格式
# ─────────────────────────────────────────────
# last_hidden_state: (B, N_spatial * T, hidden_size)
#   N_spatial = 256（每帧的空间 patch 数）
#   T = 输入帧数
#   hidden_size = 1280
z = out.last_hidden_state
print(f"\n[视频] 输入: {tuple(video.shape)}")
print(f"[视频] 输出: {tuple(z.shape)}")   # (2, 256*16, 1280) = (2, 4096, 1280)

# 如果只需要每帧的空间特征，把时间轴 reshape 出来再 squeeze
# z_per_frame: (B, T, N_spatial, hidden_size)
z_per_frame = z.view(B, T, n_patches, cfg.hidden_size)
print(f"[视频] 每帧特征: {tuple(z_per_frame.shape)}")  # (2, 16, 256, 1280)

# ─────────────────────────────────────────────
# 4. 单张图像输入
# ─────────────────────────────────────────────
# 模型本身没有"图像模式"，把同一帧复制 T 次即可。
# T=1 也可以，输出 (B, 256, 1280)，这是本项目 encoder.py 的用法。

image = torch.rand(B, C, H, W, dtype=torch.float16, device=DEVICE)

# 方式 A：T=1（本项目用法，输出直接是空间 patches）
image_input_t1 = image.unsqueeze(1)           # (B, 1, C, H, W)
with torch.no_grad():
    z_img_t1 = model(pixel_values_videos=image_input_t1).last_hidden_state
print(f"\n[图像 T=1] 输入: {tuple(image_input_t1.shape)}")
print(f"[图像 T=1] 输出: {tuple(z_img_t1.shape)}")   # (2, 256, 1280)

# 方式 B：复制 16 帧（与视频模式对齐，部分下游任务要求固定 T）
T_rep = 16
image_input_t16 = image.unsqueeze(1).expand(-1, T_rep, -1, -1, -1)  # (B, 16, C, H, W)
with torch.no_grad():
    z_img_t16 = model(pixel_values_videos=image_input_t16).last_hidden_state
print(f"\n[图像 T=16] 输入: {tuple(image_input_t16.shape)}")
print(f"[图像 T=16] 输出: {tuple(z_img_t16.shape)}")  # (2, 4096, 1280)

# ─────────────────────────────────────────────
# 5. 使用官方 AutoVideoProcessor（可选）
# ─────────────────────────────────────────────
# processor 负责 resize / normalize，输出已归一化到模型期望的值域。
# 本项目 encoder.py 手动归一化（/255），效果等价。

processor = AutoVideoProcessor.from_pretrained(CHECKPOINT_DIR)
# processor 期望输入 (T, C, H, W) uint8 numpy 或 PIL 列表
import numpy as np
raw_video = (video[0].cpu().float().numpy() * 255).astype(np.uint8)  # (T, C, H, W)
inputs = processor(raw_video, return_tensors="pt")
# inputs["pixel_values_videos"]: (1, T, C, H, W)，已归一化
print(f"\n[processor] pixel_values_videos: {tuple(inputs['pixel_values_videos'].shape)}")

print("\nDone.")
