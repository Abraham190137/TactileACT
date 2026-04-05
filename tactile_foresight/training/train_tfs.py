"""
Train Tactile Feasibility Score (TFS) — bidirectional InfoNCE contrastive learning.

Uses precomputed DINOv2 features + HDF5 actions.
Noise pretraining: actions corrupted with DDPM noise schedule (TouchGuide key trick).

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.training.train_tfs \
        --feature_dir /path/to/dino_features \
        --hdf5_dir /path/to/hdf5_episodes \
        --save_dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import math
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

from tactile_foresight.models.tfs import TactileFeasibilityScore
from tactile_foresight.datasets.tfs_dataset import TFSDataset, collate_tfs


def parse_args():
    p = argparse.ArgumentParser(description="Train TFS (contrastive learning)")
    p.add_argument("--feature_dir", type=str, required=True)
    p.add_argument("--hdf5_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--num_episodes", type=int, required=True)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--camera_names", type=str, default="global,wrist")
    p.add_argument("--action_mode", type=str, default="eef_delta",
                    choices=["eef_delta", "eef_rel", "joint_abs"],
                    help="eef_delta: 帧间差分(推荐), eef_rel: 相对首帧, joint_abs: 绝对关节角")
    p.add_argument("--pred_horizon", type=int, default=20)
    p.add_argument("--samples_per_episode", type=int, default=20)

    # Model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--init_temperature", type=float, default=0.07)

    # Noise pretraining (TouchGuide key trick)
    p.add_argument("--noise_augment", action="store_true", default=True)
    p.add_argument("--no_noise_augment", dest="noise_augment", action="store_false")
    p.add_argument("--noise_steps", type=int, default=100,
                    help="DDPM noise schedule steps (same as DP)")
    p.add_argument("--noise_ratio", type=float, default=0.5,
                    help="Fraction of batch to apply noise augmentation")

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
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


def build_noise_schedule(num_steps, device):
    """Build squaredcos_cap_v2 noise schedule (same as DP's DDPMScheduler).

    Returns alphas_cumprod: (num_steps,) tensor.
    """
    steps = num_steps + 1
    s = 0.008
    t = torch.linspace(0, num_steps, steps) / num_steps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = alphas_cumprod[1:]  # (num_steps,)
    return alphas_cumprod.to(device)


def add_action_noise(actions, alphas_cumprod, noise_ratio=0.5):
    """Add DDPM-schedule noise to a fraction of actions in the batch.

    Simulates the noisy actions that TFS will see during guided DP inference.

    Args:
        actions: (B, H, D) clean actions
        alphas_cumprod: (num_steps,) noise schedule
        noise_ratio: fraction of batch to corrupt

    Returns:
        (B, H, D) actions with noise applied to first num_noisy samples
    """
    B = actions.shape[0]
    device = actions.device
    num_noisy = int(B * noise_ratio)
    if num_noisy == 0:
        return actions

    num_steps = alphas_cumprod.shape[0]
    timesteps = torch.randint(0, num_steps, (num_noisy,), device=device)

    alpha_t = alphas_cumprod[timesteps].view(num_noisy, 1, 1)
    noise = torch.randn_like(actions[:num_noisy])
    noisy = alpha_t.sqrt() * actions[:num_noisy] + (1 - alpha_t).sqrt() * noise

    result = actions.clone()
    result[:num_noisy] = noisy
    return result


def plot_curves(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("InfoNCE Loss")
    axes[0].set_title("TFS Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Contrastive Accuracy")
    axes[1].legend()

    axes[2].plot(history["temperature"], label="logit_scale")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Logit Scale (1/tau)")
    axes[2].set_title("Learned Temperature")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "graphs", "tfs_curves.png"), dpi=100)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "graphs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_tfs] device={}".format(device))

    camera_names = args.camera_names.split(",")

    # Save config
    with open(os.path.join(args.save_dir, "tfs_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Dataset ---
    episode_ids = list(range(args.start_episode, args.start_episode + args.num_episodes))
    rng = np.random.RandomState(args.seed)
    shuffled = rng.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    train_ids = [episode_ids[i] for i in shuffled[:split]]
    val_ids = [episode_ids[i] for i in shuffled[split:]]
    print("[train_tfs] train={}, val={}".format(len(train_ids), len(val_ids)))

    train_ds = TFSDataset(
        train_ids, args.feature_dir, args.hdf5_dir, camera_names,
        pred_horizon=args.pred_horizon, action_mode=args.action_mode,
        samples_per_episode=args.samples_per_episode,
    )
    # Val uses train action stats to ensure consistent normalization
    val_ds = TFSDataset(
        val_ids, args.feature_dir, args.hdf5_dir, camera_names,
        pred_horizon=args.pred_horizon, action_mode=args.action_mode,
        samples_per_episode=args.samples_per_episode,
        action_stats=train_ds.get_action_stats(),
    )

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_tfs,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # --- Model ---
    print("[train_tfs] Building TFS (hidden={}, layers={}, heads={})...".format(
        args.hidden_dim, args.num_layers, args.nheads))
    model = TactileFeasibilityScore(
        dino_dim=768,
        action_dim=train_ds.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nheads=args.nheads,
        pred_horizon=args.pred_horizon,
        dropout=args.dropout,
        init_temperature=args.init_temperature,
        device=device,
    )
    print("[train_tfs] Trainable params: {:,}".format(model.num_trainable_params()))

    # Noise schedule for action augmentation
    alphas_cumprod = build_noise_schedule(args.noise_steps, device)
    print("[train_tfs] Noise augment: {}, steps={}, ratio={}".format(
        args.noise_augment, args.noise_steps, args.noise_ratio))

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # --- Training ---
    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "temperature": [],
    }

    for epoch in range(args.epochs):
        t0 = time.time()
        train_ds.resample()

        # --- Train ---
        model.obs_encoder.train()
        model.action_encoder.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            v_feat = batch["vision_feat"].to(device)
            tc_feat = batch["tactile_current"].to(device)
            tf_seq = batch["tactile_future_seq"].to(device)
            action = batch["action_chunk"].to(device)

            # Noise pretraining
            if args.noise_augment:
                action = add_action_noise(action, alphas_cumprod, args.noise_ratio)

            result = model.compute_loss(v_feat, tc_feat, tf_seq, action)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += result["accuracy"].item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_acc = epoch_acc / max(n_batches, 1)

        # --- Validate (clean actions) ---
        model.obs_encoder.eval()
        model.action_encoder.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                v_feat = batch["vision_feat"].to(device)
                tc_feat = batch["tactile_current"].to(device)
                tf_seq = batch["tactile_future_seq"].to(device)
                action = batch["action_chunk"].to(device)

                result = model.compute_loss(v_feat, tc_feat, tf_seq, action)
                val_loss += result["loss"].item()
                val_acc += result["accuracy"].item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_acc = val_acc / max(n_val, 1)

        cur_temp = model.temperature.item()
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)
        history["temperature"].append(cur_temp)

        dt = time.time() - t0

        if (epoch + 1) % args.log_freq == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                "[epoch {:4d}/{}] loss={:.4f}/{:.4f}  acc={:.4f}/{:.4f}  "
                "temp={:.2f}  lr={:.2e}  {:.1f}s".format(
                    epoch + 1, args.epochs,
                    avg_train_loss, avg_val_loss,
                    avg_train_acc, avg_val_acc,
                    cur_temp, lr, dt,
                )
            )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_model(os.path.join(args.save_dir, "tfs_best.pth"))

        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            model.save_model(
                os.path.join(args.save_dir, "tfs_epoch_{}.pth".format(epoch + 1))
            )

        # Plot
        if (epoch + 1) % args.plot_freq == 0:
            plot_curves(history, args.save_dir)

    # Final saves
    model.save_model(os.path.join(args.save_dir, "tfs_final.pth"))
    with open(os.path.join(args.save_dir, "tfs_history.json"), "w") as f:
        json.dump(history, f)
    plot_curves(history, args.save_dir)

    print("\n[train_tfs] Done. Best val loss: {:.4f}".format(best_val_loss))
    print("[train_tfs] Saved to {}".format(args.save_dir))


if __name__ == "__main__":
    main()
