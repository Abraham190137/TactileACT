"""
预提取所有 DINOv2 特征并缓存到磁盘。

DINOv2 frozen，同一张图片特征永远不变，没必要每个 epoch 重复计算。
预提取后训练速度可提升 10-20x。

输出格式：每条轨迹一个 .pt 文件，包含：
    {
        "vision_{cam_name}": (T, 768) float16,  # 每个摄像头
        "tactile": (T, 768) float16,
    }

Usage:
    python -m tactile_foresight.training.precompute_dino_features \
        --dataset_dir /home/chenshuai/data/dataset/260309_0310 \
        --save_dir /home/chenshuai/data/dataset/260309_0310_dino_features \
        --num_episodes 337 \
        --camera_names global,wrist \
        --tac_side left \
        --batch_size 64
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
from torchvision.transforms import Normalize
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tactile_foresight.models.feature_extractor import DINOv2Extractor


def parse_args():
    p = argparse.ArgumentParser(description="Precompute DINOv2 features")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--num_episodes", type=int, required=True)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--camera_names", type=str, default="global,wrist")
    p.add_argument("--tac_side", type=str, default="left")
    p.add_argument("--tac_key", type=str, default="img")
    p.add_argument("--batch_size", type=int, default=64,
                    help="DINOv2 推理 batch size")
    return p.parse_args()


@torch.no_grad()
def extract_features_batched(extractor, images_np, batch_size, device, normalize):
    """对一整条轨迹的图像批量提取 DINOv2 特征。

    Args:
        images_np: (T, H, W, 3) uint8
        batch_size: 推理 batch size
        device: cuda/cpu
        normalize: ImageNet normalize transform

    Returns:
        (T, 768) float16 tensor (CPU)
    """
    T = images_np.shape[0]
    all_features = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = images_np[start:end]  # (bs, H, W, 3) uint8

        # 转换: uint8 → float32 → (bs, 3, H, W) → ImageNet normalize
        imgs = torch.from_numpy(batch).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)  # (bs, 3, H, W)
        imgs = normalize(imgs)
        imgs = imgs.to(device)

        features = extractor(imgs, already_normalized=True)  # (bs, 768)
        all_features.append(features.cpu().half())

    return torch.cat(all_features, dim=0)  # (T, 768) float16


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[precompute] device={device}")

    camera_names = args.camera_names.split(",")
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("[precompute] Loading DINOv2...")
    extractor = DINOv2Extractor(device=device)

    episode_ids = list(range(args.start_episode, args.start_episode + args.num_episodes))

    for eid in tqdm(episode_ids, desc="Precompute DINOv2"):
        save_path = os.path.join(args.save_dir, f"episode_{eid}_dino.pt")
        if os.path.exists(save_path):
            continue  # 已处理，跳过

        hdf5_path = os.path.join(args.dataset_dir, f"episode_{eid}.hdf5")
        features = {}

        with h5py.File(hdf5_path, "r", swmr=True) as f:
            obs = f["observations"]

            # 视觉摄像头
            for cam_name in camera_names:
                imgs = obs["images"][cam_name][()]  # (T, H, W, 3) uint8
                feat = extract_features_batched(
                    extractor, imgs, args.batch_size, device, normalize
                )
                features[f"vision_{cam_name}"] = feat

            # 触觉
            tac_imgs = obs["tac"][args.tac_side][args.tac_key][()]  # (T, Ht, Wt, 3)
            feat = extract_features_batched(
                extractor, tac_imgs, args.batch_size, device, normalize
            )
            features["tactile"] = feat

        torch.save(features, save_path)

    print(f"\n[precompute] Done. Saved to {args.save_dir}")

    # 打印统计
    sample = torch.load(os.path.join(args.save_dir, f"episode_{episode_ids[0]}_dino.pt"))
    for k, v in sample.items():
        print(f"  {k}: {v.shape}, dtype={v.dtype}")
    file_size = os.path.getsize(os.path.join(args.save_dir, f"episode_{episode_ids[0]}_dino.pt"))
    total_size = file_size * len(episode_ids) / 1024 / 1024
    print(f"  单个文件: {file_size/1024:.1f} KB, 总计约: {total_size:.1f} MB")


if __name__ == "__main__":
    main()
