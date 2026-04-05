"""
CPM 训练数据集：加载预提取 DINOv2 特征 + HDF5 中的 action chunk。

Paper-faithful: 只需要当前 (V_t, T_t)，不需要未来触觉序列。

支持三种 action_mode：
  - "eef_delta":  eef_abs 帧间差分 (速度信号)
  - "eef_rel":    eef_abs 相对首帧 (论文默认，累计位移)
  - "joint_abs":  joint_abs 绝对值 (原始关节角)

所有模式都会做 mean/std 归一化。

每个样本:
    vision_feat: (768,) 当前视觉 DINOv2 特征
    tactile_feat: (768,) 当前触觉 DINOv2 特征
    action_chunk: (H, action_dim) 动作序列 (normalized)
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# action_mode -> (hdf5_key, needs_extra_frame)
_ACTION_MODE_CONFIG = {
    "eef_delta": ("eef_abs", True),    # 帧间差分，需要多取1帧
    "eef_rel":   ("eef_abs", False),   # 相对首帧 (论文默认)
    "joint_abs": ("joint_abs", False), # 绝对关节角
}


class CPMDataset(Dataset):
    """CPM training dataset: DINOv2 features + action chunks from HDF5.

    Unlike TFSDataset, this does NOT load future tactile sequences.
    """

    def __init__(
        self,
        episode_ids: Sequence[int],
        feature_dir: str,
        hdf5_dir: str,
        camera_names: List[str],
        pred_horizon: int = 20,
        action_mode: str = "eef_rel",
        samples_per_episode: int = 10,
        action_stats: Optional[dict] = None,
    ):
        if action_mode not in _ACTION_MODE_CONFIG:
            raise ValueError(
                "action_mode must be one of {}, got {!r}".format(
                    list(_ACTION_MODE_CONFIG.keys()), action_mode))

        self.episode_ids = list(episode_ids)
        self.feature_dir = feature_dir
        self.hdf5_dir = hdf5_dir
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.action_mode = action_mode
        self.samples_per_episode = samples_per_episode

        hdf5_key, self._needs_extra = _ACTION_MODE_CONFIG[action_mode]

        # Preload DINOv2 features
        self.all_features = {}
        for eid in episode_ids:
            path = os.path.join(feature_dir, "episode_{}_dino.pt".format(eid))
            self.all_features[eid] = torch.load(path, map_location="cpu",
                                                 weights_only=False)

        # Preload actions from HDF5
        self.all_actions = {}
        for eid in episode_ids:
            path = os.path.join(hdf5_dir, "episode_{}.hdf5".format(eid))
            with h5py.File(path, "r") as f:
                self.all_actions[eid] = torch.from_numpy(
                    f["actions/{}".format(hdf5_key)][:].astype(np.float32)
                )  # (T, action_dim)

        # Get episode length and action dim
        sample = self.all_features[episode_ids[0]]
        self.episode_length = sample["tactile"].shape[0]
        self.action_dim = self.all_actions[episode_ids[0]].shape[1]

        # max_t: ensure action chunk doesn't overflow
        extra = 1 if self._needs_extra else 0
        self.max_t = self.episode_length - self.pred_horizon - extra

        # Compute or use provided action normalization stats
        if action_stats is not None:
            self.action_mean = action_stats["mean"]
            self.action_std = action_stats["std"]
        else:
            self.action_mean, self.action_std = self._compute_action_stats()

        # Build initial index
        self._index = []
        self.resample()

        print("[CPMDataset] {} episodes, {} samples, horizon={}, action_mode={}, dim={}".format(
            len(self.episode_ids), len(self._index), pred_horizon, action_mode, self.action_dim))
        print("[CPMDataset] action mean={}, std={}".format(
            self.action_mean.numpy(), self.action_std.numpy()))

    def _compute_action_stats(self):
        """Compute mean/std of processed action chunks across all episodes."""
        all_chunks = []
        for eid in self.episode_ids:
            actions = self.all_actions[eid]  # (T, D)
            for t in range(0, self.max_t, 10):  # subsample for speed
                chunk = self._get_action_chunk(actions, t)
                all_chunks.append(chunk)
        all_chunks = torch.cat(all_chunks, dim=0)  # (N*H, D)
        mean = all_chunks.mean(dim=0)  # (D,)
        std = all_chunks.std(dim=0).clamp(min=1e-6)  # (D,)
        return mean, std

    def _get_action_chunk(self, actions, t):
        """Extract action chunk at timestep t according to action_mode.

        Returns: (H, D) tensor, un-normalized.
        """
        if self.action_mode == "eef_delta":
            # adjacent diff: delta[k] = actions[t+k+1] - actions[t+k]
            chunk_ext = actions[t: t + self.pred_horizon + 1].float()  # (H+1, D)
            return chunk_ext[1:] - chunk_ext[:-1]  # (H, D)
        elif self.action_mode == "eef_rel":
            # relative to first frame of chunk (paper Eq. 1)
            chunk = actions[t: t + self.pred_horizon].float()  # (H, D)
            return chunk - actions[t].float()  # (H, D)
        else:  # joint_abs
            return actions[t: t + self.pred_horizon].float()  # (H, D)

    def get_action_stats(self):
        """Return action stats dict for sharing between train/val datasets."""
        return {"mean": self.action_mean, "std": self.action_std}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        eid, t = self._index[idx]
        feats = self.all_features[eid]
        actions = self.all_actions[eid]

        # Random camera
        cam = random.choice(self.camera_names)
        vision_feat = feats["vision_{}".format(cam)][t].float()  # (768,)

        # Current tactile
        tactile_feat = feats["tactile"][t].float()  # (768,)

        # Action chunk: mode-specific processing + normalize
        action_chunk = self._get_action_chunk(actions, t)  # (H, D)
        action_chunk = (action_chunk - self.action_mean) / self.action_std

        return {
            "vision_feat": vision_feat,
            "tactile_feat": tactile_feat,
            "action_chunk": action_chunk,
        }

    def resample(self):
        """Resample timesteps each epoch."""
        self._index = []
        for eid in self.episode_ids:
            for _ in range(self.samples_per_episode):
                t = random.randint(0, self.max_t - 1)
                self._index.append((eid, t))


def collate_cpm(batch: list) -> dict:
    """Collate function for CPMDataset."""
    return {
        "vision_feat": torch.stack([s["vision_feat"] for s in batch]),
        "tactile_feat": torch.stack([s["tactile_feat"] for s in batch]),
        "action_chunk": torch.stack([s["action_chunk"] for s in batch]),
    }
