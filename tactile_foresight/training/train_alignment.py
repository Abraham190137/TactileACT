"""
Train DINOv2 Vision-Tactile Alignment (基于预提取特征).

DINOv2 frozen + projection heads, CLIP contrastive loss.
This is Stage 0 of TactileForesight — must be done before TFM/TFS.

前置步骤：先运行 precompute_dino_features.py 提取 DINOv2 特征到磁盘。

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.training.train_alignment \
        --feature_dir /home/chenshuai/data/dataset/260309_0310_dino_features \
        --save_dir /home/chenshuai/Project/output/vt_align \
        --num_episodes 337 \
        --camera_names global,wrist \
        --epochs 1000
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

from tactile_foresight.datasets.alignment_dataset import AlignmentFeatureDataset
from tactile_foresight.models.vt_alignment import VTAlignmentModel


def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2 VT Alignment (precomputed features)")
    p.add_argument("--feature_dir", type=str, required=True,
                    help="预提取 DINOv2 特征目录 (precompute_dino_features.py 的输出)")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--num_episodes", type=int, required=True)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--camera_names", type=str, default="global,wrist")

    # Data sampling
    p.add_argument("--n_clip_images", type=int, default=12,
                    help="Number of timesteps sampled per episode")
    p.add_argument("--min_distance", type=int, default=20,
                    help="Min timestep distance between samples")
    p.add_argument("--batch_size", type=int, default=8)

    # Model
    p.add_argument("--shared_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)

    # Training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--save_freq", type=int, default=100)
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
    """画 loss、accuracy、temperature 三张图。"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Contrastive Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Top-1 Matching Accuracy")
    axes[1].legend()

    axes[2].plot(history["temperature"], label="temperature", color="C2")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Temperature (τ)")
    axes[2].set_title("Learnable Temperature")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "graphs", "alignment_curves.png"), dpi=100)
    plt.close(fig)


def plot_similarity_map(model, vision_feats, tactile_feats, epoch, save_dir, split="val"):
    """画 (N, N) 余弦相似度热力图，直观查看对齐质量。

    Args:
        vision_feats: (N, 768) 一个 episode 第一个摄像头的 DINOv2 特征
        tactile_feats: (N, 768) 一个 episode 的触觉 DINOv2 特征
    """
    with torch.no_grad():
        v_emb = model.vision_proj(vision_feats)   # (N, shared_dim)
        t_emb = model.tactile_proj(tactile_feats)  # (N, shared_dim)
        sim = v_emb @ t_emb.T  # (N, N)
        sim = sim.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="viridis", aspect="equal", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Cosine Similarity - Epoch {epoch} ({split})")
    ax.set_xlabel("Tactile index")
    ax.set_ylabel("Vision index")
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, "graphs", f"sim_map_epoch_{epoch}_{split}.png"), dpi=100
    )
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "graphs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[alignment] device={device}")

    camera_names = args.camera_names.split(",")
    episode_ids = list(range(args.start_episode, args.start_episode + args.num_episodes))

    # Save config
    with open(os.path.join(args.save_dir, "alignment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Train/val split
    rng = np.random.RandomState(args.seed)
    shuffled = rng.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    train_ids = [episode_ids[i] for i in shuffled[:split]]
    val_ids = [episode_ids[i] for i in shuffled[split:]]
    print(f"[alignment] train={len(train_ids)}, val={len(val_ids)}")

    train_ds = AlignmentFeatureDataset(
        train_ids, args.feature_dir, camera_names,
        n_images=args.n_clip_images, min_distance=args.min_distance,
    )
    val_ds = AlignmentFeatureDataset(
        val_ids, args.feature_dir, camera_names,
        n_images=args.n_clip_images, min_distance=args.min_distance,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Build model — 不加载 DINOv2（预提取模式不需要）
    # 只构建 projection heads + temperature
    model = VTAlignmentModel(
        shared_dim=args.shared_dim,
        hidden_dim=args.hidden_dim,
        conditioning_dim=0,
        device=device,
    )
    print(f"[alignment] Trainable params: {model.num_trainable_params():,}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # Training
    best_val_loss = float("inf")
    history = {k: [] for k in [
        "train_loss", "val_loss", "train_acc", "val_acc", "temperature",
    ]}

    for epoch in range(args.epochs):
        t0 = time.time()

        # --- Train ---
        model.vision_proj.train()
        model.tactile_proj.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0

        for vision_feats, tactile_feats in train_loader:
            # vision_feats: (B, N, Cams, 768)
            # tactile_feats: (B, N, 768)
            vision_feats = vision_feats.to(device)
            tactile_feats = tactile_feats.to(device)

            result = model.compute_loss_from_features(vision_feats, tactile_feats)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += result["accuracy"]
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_acc = epoch_acc / max(n_batches, 1)

        # --- Validate ---
        model.vision_proj.eval()
        model.tactile_proj.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        val_v_feats_first = None
        val_t_feats_first = None

        with torch.no_grad():
            for batch_idx, (vision_feats, tactile_feats) in enumerate(val_loader):
                vision_feats = vision_feats.to(device)
                tactile_feats = tactile_feats.to(device)

                result = model.compute_loss_from_features(vision_feats, tactile_feats)
                val_loss += result["loss"].item()
                val_acc += result["accuracy"]
                n_val += 1

                # 记录第一个 batch 第一个 episode 用于画 similarity map
                if batch_idx == 0:
                    val_v_feats_first = vision_feats[0, :, 0]  # (N, 768) 第一个摄像头
                    val_t_feats_first = tactile_feats[0]         # (N, 768)

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_acc = val_acc / max(n_val, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)
        history["temperature"].append(model.temperature.item())

        dt = time.time() - t0

        if (epoch + 1) % args.log_freq == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            temp = model.temperature.item()
            print(
                f"[epoch {epoch+1:4d}/{args.epochs}] "
                f"loss={avg_train_loss:.4f}/{avg_val_loss:.4f}  "
                f"acc={avg_train_acc:.3f}/{avg_val_acc:.3f}  "
                f"temp={temp:.2f}  lr={lr:.2e}  {dt:.1f}s"
            )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_heads(os.path.join(args.save_dir, "alignment_best.pth"))

        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            model.save_heads(
                os.path.join(args.save_dir, f"alignment_epoch_{epoch+1}.pth")
            )

        # Plot curves + similarity map
        if (epoch + 1) % args.plot_freq == 0:
            plot_curves(history, args.save_dir)
            if val_v_feats_first is not None:
                plot_similarity_map(
                    model, val_v_feats_first, val_t_feats_first,
                    epoch + 1, args.save_dir, split="val",
                )

    # Final saves
    model.save_heads(os.path.join(args.save_dir, "alignment_final.pth"))
    with open(os.path.join(args.save_dir, "alignment_history.json"), "w") as f:
        json.dump(history, f)
    plot_curves(history, args.save_dir)

    print(f"\n[alignment] Done. Best val loss: {best_val_loss:.4f}")
    print(f"[alignment] Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
