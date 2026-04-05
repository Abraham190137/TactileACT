"""
Train Diffusion Policy (DP) — standalone training script for tactile_foresight.

Standard DDPM training: add noise -> UNet predicts noise -> MSE loss.

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.training.train_dp \
        --hdf5_dir /path/to/hdf5 \
        --save_dir /path/to/output \
        --obs_encoder dino_feature \
        --feature_dir /path/to/dino_features
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tactile_foresight.models.dp_policy import DiffusionPolicy
from tactile_foresight.datasets.dp_dataset import DPDataset, collate_dp


def parse_args():
    p = argparse.ArgumentParser(description="Train Diffusion Policy")
    # Data
    p.add_argument("--hdf5_dir", type=str, required=True)
    p.add_argument("--feature_dir", type=str, default=None,
                    help="Precomputed DINOv2 features dir (required for dino_feature mode)")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--num_episodes", type=int, required=True)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--camera_names", type=str, default="global,wrist")

    # Observation encoder
    p.add_argument("--obs_encoder", type=str, default="dino_feature",
                    choices=["dino_feature", "dino_online", "resnet18", "clip_resnet18"])
    p.add_argument("--freeze_vision", action="store_true", default=True)
    p.add_argument("--no_freeze_vision", dest="freeze_vision", action="store_false")
    p.add_argument("--clip_weights_path", type=str, default=None,
                    help="Path to CLIP pretrained weights (for clip_resnet18 mode)")

    # Tactile
    p.add_argument("--use_tactile", action="store_true", default=False,
                    help="Include tactile encoder in DP")
    p.add_argument("--clip_tac_weights_path", type=str, default=None,
                    help="Path to CLIP pretrained tactile encoder weights")
    p.add_argument("--freeze_tactile", action="store_true", default=True,
                    help="Freeze tactile backbone (default: True)")
    p.add_argument("--no_freeze_tactile", dest="freeze_tactile", action="store_false")
    p.add_argument("--tac_side", type=str, default="left",
                    choices=["left", "right"])
    p.add_argument("--tac_key", type=str, default="img",
                    choices=["img", "depth"])

    # Vision backbone sharing
    p.add_argument("--share_vision_backbone", action="store_true", default=False,
                    help="Share one vision backbone across all cameras (ACT-style)")

    # Action
    p.add_argument("--action_mode", type=str, default="eef_rel",
                    choices=["eef_delta", "eef_rel", "joint_abs"])
    p.add_argument("--pred_horizon", type=int, default=20)
    p.add_argument("--proprio_key", type=str, default="proprio_eef")
    p.add_argument("--samples_per_episode", type=int, default=50)

    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--use_ema", action="store_true", default=False)
    p.add_argument("--ema_decay", type=float, default=0.995)

    # Diffusion
    p.add_argument("--num_train_timesteps", type=int, default=100)
    p.add_argument("--num_inference_steps", type=int, default=10)

    # Logging
    p.add_argument("--save_freq", type=int, default=50)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--plot_freq", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plot_curves(history, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Diffusion Policy Training Loss")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "graphs", "dp_loss.png"), dpi=100)
    plt.close(fig)

    if history["lr"]:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
        ax2.plot(history["lr"])
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("LR Schedule")
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, "graphs", "dp_lr.png"), dpi=100)
        plt.close(fig2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "graphs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_dp] device={}".format(device))

    camera_names = args.camera_names.split(",")
    use_features = (args.obs_encoder == "dino_feature")

    if use_features and args.feature_dir is None:
        raise ValueError("--feature_dir required for dino_feature mode")

    # Save config
    with open(os.path.join(args.save_dir, "dp_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Dataset ---
    episode_ids = list(range(args.start_episode, args.start_episode + args.num_episodes))
    rng = np.random.RandomState(args.seed)
    shuffled = rng.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    train_ids = [episode_ids[i] for i in shuffled[:split]]
    val_ids = [episode_ids[i] for i in shuffled[split:]]
    print("[train_dp] train={}, val={}".format(len(train_ids), len(val_ids)))

    train_ds = DPDataset(
        train_ids, args.hdf5_dir, camera_names,
        pred_horizon=args.pred_horizon,
        action_mode=args.action_mode,
        samples_per_episode=args.samples_per_episode,
        use_features=use_features,
        feature_dir=args.feature_dir,
        proprio_key=args.proprio_key,
        use_tactile=args.use_tactile,
        tac_side=args.tac_side,
        tac_key=args.tac_key,
    )
    val_ds = DPDataset(
        val_ids, args.hdf5_dir, camera_names,
        pred_horizon=args.pred_horizon,
        action_mode=args.action_mode,
        samples_per_episode=args.samples_per_episode,
        use_features=use_features,
        feature_dir=args.feature_dir,
        proprio_key=args.proprio_key,
        action_stats=train_ds.get_action_stats(),
        proprio_stats=train_ds.get_proprio_stats(),
        use_tactile=args.use_tactile,
        tac_side=args.tac_side,
        tac_key=args.tac_key,
    )

    # Save stats
    torch.save(train_ds.get_action_stats(),
               os.path.join(args.save_dir, "action_stats.pt"))
    torch.save(train_ds.get_proprio_stats(),
               os.path.join(args.save_dir, "proprio_stats.pt"))

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_dp,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # --- Noise Scheduler ---
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # --- Model ---
    action_dim = train_ds.action_dim
    proprio_dim = train_ds.proprio_dim
    print("[train_dp] Building DP (obs_encoder={}, action_dim={}, proprio_dim={})...".format(
        args.obs_encoder, action_dim, proprio_dim))

    model = DiffusionPolicy(
        obs_encoder_type=args.obs_encoder,
        camera_names=camera_names,
        action_dim=action_dim,
        pred_horizon=args.pred_horizon,
        proprio_dim=proprio_dim,
        freeze_vision=args.freeze_vision,
        clip_weights_path=args.clip_weights_path,
        use_tactile=args.use_tactile,
        clip_tac_weights_path=args.clip_tac_weights_path,
        freeze_tactile=args.freeze_tactile,
        share_vision_backbone=args.share_vision_backbone,
        device=device,
    )
    print("[train_dp] Trainable params: {:,}".format(model.num_trainable_params()))

    # --- EMA ---
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(parameters=model.parameters(), decay=args.ema_decay)
        print("[train_dp] EMA enabled, decay={}".format(args.ema_decay))

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Training ---
    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [], "lr": [],
    }
    global_step = 0

    with tqdm(range(args.epochs), desc="Epoch") as tglobal:
        for epoch in tglobal:
            t0 = time.time()
            train_ds.resample()

            # Train
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                # Move to device
                obs = {}
                for k, v in batch.items():
                    if k != "action_chunk":
                        obs[k] = v.to(device)
                action = batch["action_chunk"].to(device)

                loss = model.compute_loss(obs, action, noise_scheduler)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                lr_scheduler.step()

                if ema_model is not None:
                    ema_model.step(model.parameters())

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                history["lr"].append(optimizer.param_groups[0]["lr"])

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    obs = {}
                    for k, v in batch.items():
                        if k != "action_chunk":
                            obs[k] = v.to(device)
                    action = batch["action_chunk"].to(device)
                    loss = model.compute_loss(obs, action, noise_scheduler)
                    val_loss += loss.item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            dt = time.time() - t0

            # Update tqdm progress bar
            tglobal.set_postfix(
                train="{:.4f}".format(avg_train_loss),
                val="{:.4f}".format(avg_val_loss),
            )

            if (epoch + 1) % args.log_freq == 0 or epoch == 0:
                lr = optimizer.param_groups[0]["lr"]
                tqdm.write("[epoch {:4d}/{}] train={:.6f}  val={:.6f}  "
                           "lr={:.2e}  {:.1f}s".format(
                               epoch + 1, args.epochs,
                               avg_train_loss, avg_val_loss, lr, dt))

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(args.save_dir, "dp_best.pth")
                model.save_model(save_path)
                if ema_model is not None:
                    ema_policy = copy.deepcopy(model)
                    ema_model.copy_to(ema_policy.parameters())
                    ema_policy.save_model(os.path.join(args.save_dir, "dp_best_ema.pth"))
                    del ema_policy

            # Periodic save
            if (epoch + 1) % args.save_freq == 0:
                model.save_model(
                    os.path.join(args.save_dir, "dp_epoch_{}.pth".format(epoch + 1)))

            # Plot
            if (epoch + 1) % args.plot_freq == 0:
                plot_curves(history, args.save_dir)

    # Final saves
    model.save_model(os.path.join(args.save_dir, "dp_final.pth"))
    if ema_model is not None:
        ema_policy = copy.deepcopy(model)
        ema_model.copy_to(ema_policy.parameters())
        ema_policy.save_model(os.path.join(args.save_dir, "dp_final_ema.pth"))
        del ema_policy

    with open(os.path.join(args.save_dir, "dp_history.json"), "w") as f:
        json.dump(history, f)
    plot_curves(history, args.save_dir)

    print("\n[train_dp] Done. Best val loss: {:.6f}".format(best_val_loss))
    print("[train_dp] Saved to {}".format(args.save_dir))


if __name__ == "__main__":
    main()
