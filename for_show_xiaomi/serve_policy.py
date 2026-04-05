"""
TactileACT TCP Policy Server (runs on GPU machine)

Usage:
    cd /path/to/TactileACT-cs
    conda activate TactileACT
    python -m for_show_xiaomi.serve_policy --ckpt_dir /path/to/ckpt_dir

ckpt_dir should contain:
    - args.json          (training config)
    - dataset_stats.pkl  (normalization stats)
    - policy_best.ckpt   (model weights)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torchvision.transforms as transforms

# Project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from policy import ACTPolicy
from utils import NormalizeSeparate, set_seed
from for_show_xiaomi.ws_server import TactileACTServer, ClientDisconnected


def build_policy(args: dict) -> ACTPolicy:
    """Build ACTPolicy from saved training args."""
    camera_names = args["camera_names"]
    state_dim = args["state_dim"]

    pretrained_backbones = None
    camera_backbone_mapping = None

    if args.get("backbone") == "clip_backbone":
        from clip_pretraining_xiaomi import modified_resnet18
        vision_model = modified_resnet18()
        gelsight_model = modified_resnet18()
        camera_backbone_mapping = {c: 0 for c in camera_names}
        camera_backbone_mapping["gelsight"] = 1
        pretrained_backbones = [vision_model, gelsight_model]

    return ACTPolicy(
        state_dim=state_dim,
        hidden_dim=args.get("hidden_dim", 512),
        position_embedding_type=args.get("position_embedding", "sine"),
        lr_backbone=args.get("lr_backbone", 1e-5),
        masks=args.get("masks", False),
        backbone_type=args.get("backbone", "resnet18"),
        dilation=args.get("dilation", False),
        dropout=args.get("dropout", 0.1),
        nheads=args.get("nheads", 8),
        dim_feedforward=args.get("dim_feedforward", 2048),
        num_enc_layers=args.get("enc_layers", 4),
        num_dec_layers=args.get("dec_layers", 7),
        pre_norm=args.get("pre_norm", False),
        num_queries=args.get("chunk_size", 20),
        camera_names=camera_names,
        z_dimension=args.get("z_dimension", 32),
        lr=args.get("lr", 1e-5),
        weight_decay=args.get("weight_decay", 1e-4),
        kl_weight=args.get("kl_weight", 10),
        pretrained_backbones=pretrained_backbones,
        cam_backbone_mapping=camera_backbone_mapping,
    )


_IMG_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


def preprocess_images(obs: dict, camera_names: list,
                      norm_stats: dict, device: torch.device) -> list:
    """
    Convert raw obs into list of image tensors matching ACTPolicy input.

    obs format from client:
      obs["images"][cam_name] -> (H,W,3) uint8
      obs["tac"]["left"]["img"] -> (H,W,3) uint8   (maps to "gelsight" camera)
    """
    all_images = []
    for cam_name in camera_names:
        if cam_name == "gelsight":
            # --- tactile ---
            if "tac" in obs:
                tac = obs["tac"]
                side = list(tac.keys())[0]
                side_data = tac[side]
                img = side_data["img"] if isinstance(side_data, dict) else side_data
                img = np.asarray(img, dtype=np.float32)
                if img.max() > 1.0:
                    img = img / 255.0
                t = torch.from_numpy(img).permute(2, 0, 1).float()
                t = _IMG_NORM(t)
            elif "gelsight" in obs:
                gs = np.asarray(obs["gelsight"], dtype=np.float32)
                gs_mean = norm_stats.get("gelsight_mean")
                gs_std = norm_stats.get("gelsight_std")
                if gs_mean is not None:
                    gs = (gs - gs_mean) / gs_std
                t = torch.from_numpy(gs).permute(2, 0, 1).float()
            else:
                raise KeyError(f"No tactile data for 'gelsight' camera. obs keys: {list(obs.keys())}")
            all_images.append(t.unsqueeze(0).to(device))

        elif cam_name == "blank":
            all_images.append(torch.zeros(1, 3, 480, 640, device=device))

        else:
            # --- regular camera ---
            if "images" in obs and cam_name in obs["images"]:
                img = obs["images"][cam_name]
            elif cam_name in obs:
                img = obs[cam_name]
            else:
                raise KeyError(f"Camera '{cam_name}' not in obs")
            img = np.asarray(img, dtype=np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).float()
            t = _IMG_NORM(t)
            all_images.append(t.unsqueeze(0).to(device))

    return all_images


def main():
    parser = argparse.ArgumentParser(description="TactileACT Policy Server")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_best.ckpt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server] device: {device}")

    # --- load training config ---
    args_path = os.path.join(cli.ckpt_dir, "args.json")
    with open(args_path) as f:
        train_args = json.load(f)

    stats_path = os.path.join(cli.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    # norm_stats embedded in args.json (as lists) -> convert to numpy
    if "norm_stats" in train_args:
        for k, v in train_args["norm_stats"].items():
            if k not in norm_stats:
                norm_stats[k] = np.array(v)

    normalizer = NormalizeSeparate(norm_stats)
    camera_names = train_args["camera_names"]
    state_dim = train_args["state_dim"]
    chunk_size = train_args["chunk_size"]
    temporal_agg = cli.temporal_agg or train_args.get("temporal_agg", False)

    print(f"[server] cameras={camera_names}  state_dim={state_dim}  "
          f"chunk={chunk_size}  temporal_agg={temporal_agg}")

    # --- build & load model ---
    policy = build_policy(train_args)
    ckpt_path = os.path.join(cli.ckpt_dir, cli.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    result = policy.load_state_dict(ckpt, strict=False)
    if result.missing_keys:
        print(f"  warning missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  warning unexpected keys: {result.unexpected_keys}")
    policy.to(device)
    policy.eval()
    print(f"[server] loaded {ckpt_path}")

    query_freq = 1 if temporal_agg else chunk_size

    # --- start TCP server ---
    server = TactileACTServer(
        host=cli.host, port=cli.port,
        metadata={"protocol": "tactileact", "camera_names": camera_names,
                   "state_dim": state_dim, "chunk_size": chunk_size,
                   "temporal_agg": temporal_agg},
    )
    server.start()
    print(f"[server] listening on {cli.host}:{cli.port}  (waiting for client...)")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[server] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                print("[server] client gone before first obs, waiting...")
                continue

            if temporal_agg:
                all_time_actions = torch.zeros(
                    [cli.max_timesteps, cli.max_timesteps + chunk_size, state_dim],
                    device=device)

            with torch.inference_mode():
                try:
                    for t in range(cli.max_timesteps):
                        # preprocess
                        qpos_raw = np.asarray(obs["qpos"], dtype=np.float32)    #原始关节角
                        qpos_n = normalizer.normalize_qpos(qpos_raw)    #归一化
                        qpos_t = torch.from_numpy(qpos_n).float().unsqueeze(0).to(device)
                        imgs = preprocess_images(obs, camera_names, norm_stats, device)

                        # inference
                        if t % query_freq == 0:
                            all_actions = policy(qpos_t, imgs)  # (1, chunk, dim)


                        if temporal_agg:
                            all_time_actions[t, t:t+chunk_size] = all_actions.squeeze(0)
                            col = all_time_actions[:, t]
                            mask = torch.all(col != 0, dim=1)
                            col = col[mask]
                            k = 0.9  # 衰减系数、越大越灵敏，越信任最新预测，ACT原版是0.01（非常平滑，但快速反映的任务可能是卡住）
                            w = np.exp(-k * np.arange(len(col)))  # 最新预测权重最大，越旧的预测越小
                            w = w / w.sum()
                            w = torch.from_numpy(w).to(device).unsqueeze(1).float()
                            raw = (col * w).sum(dim=0, keepdim=True)
                        else:
                            raw = all_actions[:, t % query_freq]

                        raw_np = raw.squeeze(0).cpu().numpy()

                        # un-normalize → absolute action (joint angles)
                        action = normalizer.unnormalize_action(raw_np)
                        action = action.astype(np.float32)

                        server.send_action({
                            "actions": action[None, :],
                            "step": t,
                        })

                        if t + 1 < cli.max_timesteps:
                            obs = server.recv_obs()

                except ClientDisconnected:
                    print(f"[server] client disconnected at step {t}")

            ep += 1
    except KeyboardInterrupt:
        print("\n[server] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
