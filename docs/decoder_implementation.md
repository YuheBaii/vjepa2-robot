# MLP Decoder 实现文档

**日期**: 2026-05-19
**任务**: 为 pick_and_lift 任务构建 RLBench 数据加载器 + Attention MLP Decoder + 训练脚本

---

## 1. 项目背景

- JEPA encoder: `facebook/vjepa2-vith-fpc64-256` (ViT-Huge)
- Encoder 输出维度: `(B, 256, 1280)` — 256 patches × 1280 embed_dim
- Encoder 在训练中完全冻结，只训练 Decoder

## 2. 新增文件清单

| 文件 | 用途 |
|------|------|
| `src/data/rlbench_dataset.py` | RLBench HDF5 数据加载器 |
| `src/decoder/__init__.py` | Decoder 模块导出 |
| `src/decoder/mlp_decoder.py` | MLP Decoder 网络定义（含 AttentionPool） |
| `scripts/collect_rlbench_data.py` | RLBench 数据采集脚本（live_demos） |
| `scripts/train_decoder.py` | 训练脚本（冻结 encoder，训练 decoder） |
| `configs/decoder.yaml` | Decoder 训练超参数配置 |

### 已有文件修改

| 文件 | 修改内容 |
|------|----------|
| `src/data/__init__.py` | 新增 `RLBenchDataset` 导出 |

## 3. 数据流

```
RLBench PickAndLift  →  HDF5 文件 (collect_rlbench_data.py)
  ├─ front_rgb     (N, 256, 256, 3) uint8
  ├─ ee_pos        (N, 3)  float32    末端执行器世界坐标
  └─ contact       (N,)    float32    0=未接触, 1=已接触(gripper closed)

           ↓  RLBenchDataset.__getitem__

  image (3,256,256) + ee_pos (3,) + contact (scalar)

           ↓  Frozen VJEPA2Encoder.forward()

  z_t  (B, 256, 1280)

           ↓  MLPDecoder.forward()

  AttentionPool(patches) → trunk (Linear→ReLU)×3 → ┬─ ee_head:      Linear(512→3)   MSE loss
                                                     └─ contact_head: Linear(512→1)   BCEWithLogits loss
```

## 4. MLP Decoder 架构细节

**文件**: `src/decoder/mlp_decoder.py`

### AttentionPool

使用可学习的 query token 对 patch embeddings 做注意力池化，替代简单的 mean pooling：

```
输入: x (B, N_patches, embed_dim)
  query: 可学习参数 (1, 1, embed_dim)，初始化为 trunc_normal(std=0.02)
  key:   Linear(embed_dim, embed_dim, bias=False) 投影
  attn = softmax(query @ key^T / sqrt(embed_dim))        # (B, 1, N)
  z = attn @ x                                            # (B, 1, embed_dim)
输出: (B, embed_dim)
```

### 整体结构

```
输入: z_t (B, 256, 1280)
  ↓ AttentionPool (learnable query token attends to patches)
(B, 1280)
  ↓ Linear(1280, 512) → ReLU
  ↓ Linear(512, 512) → ReLU
  ↓ Linear(512, 512) → ReLU
(B, 512)
  ├─→ Linear(512, 3)  → ee_pos (B, 3)
  └─→ Linear(512, 1)  → contact_logit (B, 1)
```

- 总参数量: ~2M (仅 decoder)
- 两个 head 独立，共享 trunk
- contact 输出为 raw logit，配合 `BCEWithLogitsLoss` 使用（数值更稳定）

## 5. Loss 函数

| 分支 | Loss | 权重 (config) |
|------|------|---------------|
| ee_pos | MSE | 1.0 |
| contact | BCEWithLogitsLoss | 0.5 |

```
total_loss = 1.0 * MSE(ee_pred, ee_gt)
           + 0.5 * BCE(contact_logit, contact_gt)
```

## 6. 评估指标 (每个 epoch 记录)

| 指标 | 含义 | 记录到 wandb |
|------|------|-------------|
| `loss/total` | 加权总 loss | train + val |
| `loss/ee_pos` | EE 位置 MSE | train + val |
| `loss/contact` | Contact BCE | train + val |
| `error/ee_pos_l2` | EE 位置 L2 误差 (米) | train + val |
| `error/contact_acc` | Contact 分类准确率 | train + val |

## 7. 训练配置

**文件**: `configs/decoder.yaml`

```yaml
encoder:
  checkpoint_dir: checkpoints
  dtype: float16

decoder:
  embed_dim: 1280
  hidden_dim: 512
  num_layers: 3

data:
  h5_path: data/rlbench_pick_and_lift.h5
  train_ratio: 0.8

training:
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  epochs: 50
  num_workers: 4
  prefetch_factor: 2

loss:
  ee_pos_weight: 1.0
  contact_weight: 0.5
```

- 优化器: AdamW
- 学习率调度: CosineAnnealingLR (T_max = epochs)
- 数据划分: 80% 训练 / 20% 验证

## 8. 使用方式

```bash
# Step 1: 采集 RLBench 演示数据
python scripts/collect_rlbench_data.py \
    --num_demos 50 \
    --output data/rlbench_pick_and_lift.h5

# Step 2: 训练 Decoder
python scripts/train_decoder.py --config configs/decoder.yaml

# 不使用 wandb:
python scripts/train_decoder.py --config configs/decoder.yaml --no_wandb

# 指定 GPU:
python scripts/train_decoder.py --config configs/decoder.yaml --device cuda:0
```

## 9. Checkpoint 保存

保存在 `checkpoints/decoder/`:
- `best_decoder.pt` — 验证 loss 最低的 checkpoint（含 optimizer state、epoch、val_metrics）
- `final_decoder.pt` — 最后一个 epoch 的 checkpoint

## 10. 数据采集注意事项

- RLBench 需要 CoppeliaSim 仿真器运行中
- 使用 `live_demos=True` 实时生成演示（非预录制）
- 使用 `_step_callback` 在每个 step 记录 front_rgb、ee_pos、contact
- `contact = 1.0 - gripper_open` (1=夹爪闭合接触物体, 0=夹爪张开未接触)
- 如果某个 variation 执行失败（RuntimeError），跳过并继续下一个
- 数据保存在 HDF5 格式，front_rgb 使用 gzip 压缩
