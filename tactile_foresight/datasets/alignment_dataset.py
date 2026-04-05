"""
基于预提取 DINOv2 特征的对齐数据集。

直接加载 768 维向量，不需要在训练时过 DINOv2，速度提升 10-20x。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _sample_spaced_timesteps(T: int, n: int, min_distance: int) -> List[int]:
    """从长度 T 的轨迹中采样 n 个间隔 >= min_distance 的时间步。"""
    segment_len = T / n
    timesteps = []
    for i in range(n):
        lo = int(i * segment_len)
        hi = int((i + 1) * segment_len)
        lo = max(lo, 0)
        hi = min(hi, T)
        t = int(np.random.randint(lo, hi))
        timesteps.append(t)
    timesteps.sort()
    valid = all(timesteps[j] - timesteps[j - 1] >= min_distance for j in range(1, len(timesteps)))
    if valid:
        return timesteps
    # fallback
    timesteps = []
    available = T - (n - 1) * min_distance
    if available <= 0:
        return [int(i * min_distance) for i in range(n)]
    offsets = sorted(np.random.choice(available, size=n, replace=False))
    for i, off in enumerate(offsets):
        timesteps.append(int(off + i * min_distance))
    return timesteps


class AlignmentFeatureDataset(Dataset):
    """加载预提取的 DINOv2 特征，用于对齐训练。

    每个 item 返回：
        vision_feats: (N, Cams, 768) float32
        tactile_feats: (N, 768) float32
    """

    def __init__(
        self,
        episode_ids: List[int],
        feature_dir: str,
        camera_names: List[str],
        n_images: int = 12,
        min_distance: int = 20,
    ):
        self.episode_ids = episode_ids
        self.feature_dir = feature_dir
        self.camera_names = camera_names
        self.n_images = n_images
        self.min_distance = min_distance

        # 预加载所有特征到内存（float16 很小，337条×300步×3流×768维 ≈ 440MB）
        self.all_features = {}
        for eid in episode_ids:
            path = f"{feature_dir}/episode_{eid}_dino.pt"
            self.all_features[eid] = torch.load(path, map_location="cpu")

        # 获取轨迹长度
        sample = self.all_features[episode_ids[0]]
        self.episode_length = sample["tactile"].shape[0]

        assert self.episode_length >= n_images * min_distance * 1.2, (
            f"轨迹太短: length={self.episode_length}, n_images={n_images}, min_distance={min_distance}"
        )

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int):
        eid = self.episode_ids[index]
        feats = self.all_features[eid]

        timesteps = _sample_spaced_timesteps(self.episode_length, self.n_images, self.min_distance)

        # vision: (N, Cams, 768)
        cam_feats = []
        for cam_name in self.camera_names:
            cam_feats.append(feats[f"vision_{cam_name}"][timesteps].float())  # (N, 768)
        vision = torch.stack(cam_feats, dim=1)  # (N, Cams, 768)

        # tactile: (N, 768)
        tactile = feats["tactile"][timesteps].float()  # (N, 768)

        return vision, tactile
