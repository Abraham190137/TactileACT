"""
Train the Tactile Foresight Model (TFM) v2 — 序列预测版.

预测未来 H 步触觉序列，直接在 DINOv2 768-dim 空间。

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.training.train_tfm \
        --feature_dir /home/chenshuai/data/dataset/260309_0310_dino_features \
        --save_dir /home/chenshuai/Project/output/tfm_xiaomi_seq20 \
        --num_episodes 337 \
        --pred_horizon 20 \
        --epochs 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tactile_foresight.models.tfm import TactileForesightModel
from tactile_foresight.datasets.foresight_feature_dataset import (
    ForesightFeatureDataset,
    collate_foresight_features,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train TFM v2 (sequence prediction)")
    p.add_argument("--feature_dir", type=str, required=True,
                    help="预提取 DINOv2 特征目录")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--num_episodes", type=int, required=True)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--camera_names", type=str, default="global,wrist")
    p.add_argument("--pred_horizon", type=int, default=20,
                    help="预测未来多少步触觉序列")
    p.add_argument("--tac_mask_ratio", type=float, default=0.5,
                    help="训练时随机 mask 当前触觉的比例")
    p.add_argument("--samples_per_episode", type=int, default=10,
                    help="每条轨迹每个 epoch 采样的样本数")

    # Model
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=50)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--plot_freq", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_curves(history, save_dir):
    """画 loss 和 cosine similarity 曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("TFM v2 Loss (sequence prediction)")
    axes[0].legend()

    axes[1].plot(history["train_cosine_sim"], label="train")
    axes[1].plot(history["val_cosine_sim"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Pred vs Target Cosine Similarity (avg over H steps)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "graphs", "tfm_curves.png"), dpi=100)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "graphs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_tfm] device={device}")

    camera_names = args.camera_names.split(",")

    # Save config
    with open(os.path.join(args.save_dir, "tfm_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Dataset ---
    episode_ids = list(range(args.start_episode, args.start_episode + args.num_episodes))

    # Train/val split
    rng = np.random.RandomState(args.seed)
    shuffled = rng.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    train_ids = [episode_ids[i] for i in shuffled[:split]]
    val_ids = [episode_ids[i] for i in shuffled[split:]]
    print(f"[train_tfm] train={len(train_ids)}, val={len(val_ids)}")

    train_ds = ForesightFeatureDataset(
        train_ids, args.feature_dir, camera_names,
        pred_horizon=args.pred_horizon, tac_mask_ratio=args.tac_mask_ratio,
        samples_per_episode=args.samples_per_episode,
    )
    val_ds = ForesightFeatureDataset(
        val_ids, args.feature_dir, camera_names,
        pred_horizon=args.pred_horizon, tac_mask_ratio=args.tac_mask_ratio,
        samples_per_episode=args.samples_per_episode,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_foresight_features,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # --- Model ---
    print(f"[train_tfm] Building TFM v2 (pred_horizon={args.pred_horizon})...")
    model = TactileForesightModel(
        dino_dim=768,
        pred_horizon=args.pred_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nheads=args.nheads,
        dropout=args.dropout,
        device=device,
    )
    print(f"[train_tfm] Trainable params: {model.num_trainable_params():,}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # --- Training loop ---
    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_cosine_sim": [], "val_cosine_sim": [],
    }

    for epoch in range(args.epochs):
        t0 = time.time()

        # 每个 epoch 重新采样时间步
        train_ds.resample()

        # --- Train ---
        model.pred_head.train()
        epoch_loss, epoch_cos, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            v_feat = batch["vision_feat"].to(device)
            t_cur = batch["tac_current_feat"].to(device)
            t_fut_seq = batch["tac_future_seq"].to(device)
            tac_mask = batch["tac_mask"].to(device)

            result = model.compute_loss(v_feat, t_cur, t_fut_seq, tac_mask)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cos += result["cosine_sim"].item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_cos = epoch_cos / max(n_batches, 1)

        # --- Validate ---
        model.pred_head.eval()
        val_loss, val_cos, n_val = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                v_feat = batch["vision_feat"].to(device)
                t_cur = batch["tac_current_feat"].to(device)
                t_fut_seq = batch["tac_future_seq"].to(device)
                tac_mask = batch["tac_mask"].to(device)

                result = model.compute_loss(v_feat, t_cur, t_fut_seq, tac_mask)
                val_loss += result["mse_loss"].item()
                val_cos += result["cosine_sim"].item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_cos = val_cos / max(n_val, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_cosine_sim"].append(avg_train_cos)
        history["val_cosine_sim"].append(avg_val_cos)

        dt = time.time() - t0

        if (epoch + 1) % args.log_freq == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[epoch {epoch+1:4d}/{args.epochs}] "
                f"loss={avg_train_loss:.4f}/{avg_val_loss:.4f}  "
                f"cos_sim={avg_train_cos:.4f}/{avg_val_cos:.4f}  "
                f"lr={lr:.2e}  {dt:.1f}s"
            )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_model(os.path.join(args.save_dir, "tfm_best.pth"))

        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            model.save_model(os.path.join(args.save_dir, f"tfm_epoch_{epoch+1}.pth"))

        # Plot
        if (epoch + 1) % args.plot_freq == 0:
            plot_curves(history, args.save_dir)

    # Final saves
    model.save_model(os.path.join(args.save_dir, "tfm_final.pth"))
    with open(os.path.join(args.save_dir, "tfm_history.json"), "w") as f:
        json.dump(history, f)
    plot_curves(history, args.save_dir)

    print(f"\n[train_tfm] Done. Best val loss: {best_val_loss:.4f}")
    print(f"[train_tfm] Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
