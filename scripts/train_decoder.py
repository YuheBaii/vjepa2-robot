"""Train MLP Decoder on RLBench pick_and_lift data with frozen JEPA encoder.

Usage:
    python scripts/train_decoder.py --config configs/decoder.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoder.encoder import VJEPA2Encoder
from src.decoder.mlp_decoder import MLPDecoder
from src.data.rlbench_dataset import RLBenchDataset


def _compute_metrics(
    ee_pred: torch.Tensor,
    contact_logit: torch.Tensor,
    ee_gt: torch.Tensor,
    contact_gt: torch.Tensor,
    loss_weights: dict[str, float],
):
    """Compute losses and prediction errors for EE position and contact."""
    mse = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    loss_ee = mse(ee_pred, ee_gt) * loss_weights["ee_pos_weight"]
    loss_contact = (
        bce_logits(contact_logit.squeeze(-1), contact_gt)
        * loss_weights["contact_weight"]
    )
    total_loss = loss_ee + loss_contact

    with torch.no_grad():
        ee_error = torch.norm(ee_pred - ee_gt, dim=-1).mean()
        contact_pred = (torch.sigmoid(contact_logit.squeeze(-1)) > 0.5).float()
        contact_acc = (contact_pred == contact_gt).float().mean()

    return {
        "loss/total": total_loss.item(),
        "loss/ee_pos": loss_ee.item(),
        "loss/contact": loss_contact.item(),
        "error/ee_pos_l2": ee_error.item(),
        "error/contact_acc": contact_acc.item(),
    }, total_loss


def train_one_epoch(
    encoder: VJEPA2Encoder,
    decoder: MLPDecoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: dict[str, float],
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    is_train: bool = True,
) -> dict[str, float]:
    """Run one epoch of training or validation."""
    if is_train:
        decoder.train()
    else:
        decoder.eval()

    prefix = "train" if is_train else "val"
    epoch_metrics: dict[str, float] = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d} {prefix}", leave=False)
    for batch in pbar:
        images, ee_pos, contact = [x.to(device) for x in batch]

        with torch.no_grad():
            z_t = encoder(images).float()

        ee_pred, contact_logit = decoder(z_t)

        metrics, loss = _compute_metrics(
            ee_pred, contact_logit, ee_pos, contact, loss_weights,
        )

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        for k, v in metrics.items():
            epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
        num_batches += 1

        pbar.set_postfix(
            loss=metrics["loss/total"],
            ee=metrics["error/ee_pos_l2"],
            contact=metrics["error/contact_acc"],
        )

    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches

    if use_wandb:
        import wandb
        wandb.log(
            {f"{prefix}/{k.split('/')[-1]}": v for k, v in epoch_metrics.items()},
            step=epoch,
        )

    return epoch_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP Decoder for pick_and_lift")
    parser.add_argument("--config", type=str, default="configs/decoder.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/decoder")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_wandb = not args.no_wandb

    if use_wandb:
        import wandb
        wandb.init(project="vjepa2-robot-decoder", config=cfg)

    # --- Encoder (frozen) ---
    print("Loading V-JEPA 2 encoder (frozen)...")
    encoder = VJEPA2Encoder(
        checkpoint_dir=cfg["encoder"]["checkpoint_dir"],
        dtype=getattr(torch, cfg["encoder"]["dtype"]),
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # --- Decoder (trainable) ---
    decoder = MLPDecoder(
        embed_dim=cfg["decoder"]["embed_dim"],
        hidden_dim=cfg["decoder"]["hidden_dim"],
        num_layers=cfg["decoder"]["num_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {total_params:,}")

    # --- Data ---
    dataset = RLBenchDataset(h5_path=cfg["data"]["h5_path"])
    train_ratio = cfg["data"]["train_ratio"]
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    print(f"Train frames: {train_len}, Val frames: {val_len}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        prefetch_factor=cfg["training"]["prefetch_factor"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        prefetch_factor=cfg["training"]["prefetch_factor"],
        pin_memory=True,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    loss_weights = cfg["loss"]
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            encoder, decoder, train_loader, optimizer, loss_weights,
            device, epoch, use_wandb, is_train=True,
        )
        scheduler.step()

        val_metrics = train_one_epoch(
            encoder, decoder, val_loader, None, loss_weights,
            device, epoch, use_wandb, is_train=False,
        )

        t_loss = train_metrics["loss/total"]
        v_loss = val_metrics["loss/total"]
        v_ee_err = val_metrics["error/ee_pos_l2"]
        v_contact_acc = val_metrics["error/contact_acc"]

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss: {t_loss:.4f} | "
            f"val_loss: {v_loss:.4f} | "
            f"ee_l2: {v_ee_err:.4f}m | "
            f"contact_acc: {v_contact_acc:.3f}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(
                {
                    "epoch": epoch,
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": v_loss,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.checkpoint_dir, "best_decoder.pt"),
            )
            print(f"  -> saved best checkpoint (val_loss={v_loss:.4f})")

    torch.save(
        {
            "epoch": cfg["training"]["epochs"],
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(args.checkpoint_dir, "final_decoder.pt"),
    )
    print(f"Training complete. Checkpoints saved to {args.checkpoint_dir}")

    dataset.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
