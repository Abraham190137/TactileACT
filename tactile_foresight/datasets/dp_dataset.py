"""
Diffusion Policy training dataset.

Supports two data loading modes:
  - Feature mode (dino_feature): load precomputed DINOv2 .pt features
  - Image mode (dino_online / resnet18 / clip_resnet18): load raw images from HDF5

Supports three action modes (same as CPM):
  - eef_rel:   relative to current timestep t (paper default, 6D)
  - eef_delta:  frame-to-frame diff (6D)
  - joint_abs:  absolute joint angles (7D)

Each sample returns:
  - vision_{cam}: (768,) feature or (3, 224, 224) image per camera
  - tactile: (768,) feature or (3, 224, 224) tactile image (if use_tactile)
  - proprio: (proprio_dim,) normalized proprioceptive state
  - action_chunk: (H, action_dim) normalized action sequence
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

_ACTION_MODE_CONFIG = {
    "eef_delta": ("eef_abs", True),    # frame diff, needs extra frame
    "eef_rel":   ("eef_abs", False),   # relative to current timestep t
    "joint_abs": ("joint_abs", False), # absolute joint angles
}


class DPDataset(Dataset):
    """Diffusion Policy training dataset.

    Args:
        episode_ids: list of episode indices
        hdf5_dir: directory containing episode_*.hdf5 files
        camera_names: list of camera names
        pred_horizon: action chunk length
        action_mode: one of eef_rel / eef_delta / joint_abs
        samples_per_episode: number of random samples per episode per epoch
        use_features: if True, load precomputed DINOv2 features (dino_feature mode)
        feature_dir: directory containing precomputed DINOv2 .pt files
        proprio_key: HDF5 key for proprioceptive data
        action_stats: optional precomputed action normalization stats
        proprio_stats: optional precomputed proprio normalization stats
        already_normalized: if True, images are already ImageNet-normalized
    """

    def __init__(
        self,
        episode_ids: Sequence[int],
        hdf5_dir: str,
        camera_names: List[str],
        pred_horizon: int = 20,
        action_mode: str = "eef_rel",
        samples_per_episode: int = 50,
        use_features: bool = True,
        feature_dir: Optional[str] = None,
        proprio_key: str = "proprio_eef",
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
        proprio_stats: Optional[Dict[str, torch.Tensor]] = None,
        already_normalized: bool = True,
        use_tactile: bool = False,
        tac_side: str = "left",
        tac_key: str = "img",
    ):
        if action_mode not in _ACTION_MODE_CONFIG:
            raise ValueError("action_mode must be one of {}, got {!r}".format(
                list(_ACTION_MODE_CONFIG.keys()), action_mode))

        self.episode_ids = list(episode_ids)
        self.hdf5_dir = hdf5_dir
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.action_mode = action_mode
        self.samples_per_episode = samples_per_episode
        self.use_features = use_features
        self.feature_dir = feature_dir
        self.proprio_key = proprio_key
        self.already_normalized = already_normalized
        self.use_tactile = use_tactile
        self.tac_side = tac_side
        self.tac_key = tac_key

        hdf5_key, self._needs_extra = _ACTION_MODE_CONFIG[action_mode]

        # Preload actions and proprio from HDF5
        self.all_actions = {}
        self.all_proprio = {}
        for eid in episode_ids:
            path = os.path.join(hdf5_dir, "episode_{}.hdf5".format(eid))
            with h5py.File(path, "r") as f:
                self.all_actions[eid] = torch.from_numpy(
                    f["actions/{}".format(hdf5_key)][:].astype(np.float32))
                self.all_proprio[eid] = torch.from_numpy(
                    f["observations/{}".format(proprio_key)][:].astype(np.float32))

        # Preload features or prepare image loading
        if use_features:
            assert feature_dir is not None, "feature_dir required for feature mode"
            self.all_features = {}
            for eid in episode_ids:
                path = os.path.join(feature_dir, "episode_{}_dino.pt".format(eid))
                self.all_features[eid] = torch.load(
                    path, map_location="cpu", weights_only=False)
        else:
            self.all_features = None
            # Image transform: resize to 224x224, add ImageNet normalize if needed
            transforms = [
                T.Resize(224, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.CenterCrop(224),
            ]
            if not already_normalized:
                transforms.append(T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ))
            self._img_transform = T.Compose(transforms)

        # Dimensions
        sample_eid = episode_ids[0]
        self.episode_length = self.all_actions[sample_eid].shape[0]
        self.action_dim = self.all_actions[sample_eid].shape[1]
        self.proprio_dim = self.all_proprio[sample_eid].shape[1]

        # max_t: ensure action chunk doesn't overflow
        extra = 1 if self._needs_extra else 0
        self.max_t = self.episode_length - self.pred_horizon - extra

        # Compute or use provided normalization stats
        if action_stats is not None:
            self.action_mean = action_stats["mean"]
            self.action_std = action_stats["std"]
        else:
            self.action_mean, self.action_std = self._compute_action_stats()

        if proprio_stats is not None:
            self.proprio_mean = proprio_stats["mean"]
            self.proprio_std = proprio_stats["std"]
        else:
            self.proprio_mean, self.proprio_std = self._compute_proprio_stats()

        # Build initial index
        self._index = []
        self.resample()

        print("[DPDataset] {} episodes, {} samples, horizon={}, action_mode={}, "
              "dim={}, proprio_dim={}, use_features={}".format(
                  len(self.episode_ids), len(self._index), pred_horizon,
                  action_mode, self.action_dim, self.proprio_dim, use_features))

    def _get_action_chunk(self, actions, t):
        """Extract action chunk at timestep t, with padding if needed."""
        T_total = actions.shape[0]
        if self.action_mode == "eef_delta":
            end = min(t + self.pred_horizon + 1, T_total)
            chunk_ext = actions[t:end].float()
            delta = chunk_ext[1:] - chunk_ext[:-1]
            # Pad if needed
            if delta.shape[0] < self.pred_horizon:
                pad = delta[-1:].expand(self.pred_horizon - delta.shape[0], -1)
                delta = torch.cat([delta, pad], dim=0)
            return delta
        elif self.action_mode == "eef_rel":
            end = min(t + self.pred_horizon, T_total)
            chunk = actions[t:end].float()
            rel = chunk - actions[t].float()
            if rel.shape[0] < self.pred_horizon:
                pad = rel[-1:].expand(self.pred_horizon - rel.shape[0], -1)
                rel = torch.cat([rel, pad], dim=0)
            return rel
        else:  # joint_abs
            end = min(t + self.pred_horizon, T_total)
            chunk = actions[t:end].float()
            if chunk.shape[0] < self.pred_horizon:
                pad = chunk[-1:].expand(self.pred_horizon - chunk.shape[0], -1)
                chunk = torch.cat([chunk, pad], dim=0)
            return chunk

    def _compute_action_stats(self):
        all_chunks = []
        for eid in self.episode_ids:
            actions = self.all_actions[eid]
            for t in range(0, self.max_t, 10):
                chunk = self._get_action_chunk(actions, t)
                all_chunks.append(chunk)
        all_chunks = torch.cat(all_chunks, dim=0)
        mean = all_chunks.mean(dim=0)
        std = all_chunks.std(dim=0).clamp(min=1e-6)
        return mean, std

    def _compute_proprio_stats(self):
        all_proprio = []
        for eid in self.episode_ids:
            all_proprio.append(self.all_proprio[eid])
        all_proprio = torch.cat(all_proprio, dim=0)
        mean = all_proprio.mean(dim=0)
        std = all_proprio.std(dim=0).clamp(min=1e-6)
        return mean, std

    def get_action_stats(self):
        return {"mean": self.action_mean, "std": self.action_std}

    def get_proprio_stats(self):
        return {"mean": self.proprio_mean, "std": self.proprio_std}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        eid, t = self._index[idx]
        actions = self.all_actions[eid]
        proprio = self.all_proprio[eid]

        result = {}

        # Vision features or images
        if self.use_features:
            feats = self.all_features[eid]
            for cam in self.camera_names:
                result["vision_{}".format(cam)] = feats["vision_{}".format(cam)][t].float()
            # Tactile from precomputed features
            if self.use_tactile:
                result["tactile"] = feats["tactile"][t].float()
        else:
            # Load images from HDF5
            path = os.path.join(self.hdf5_dir, "episode_{}.hdf5".format(eid))
            with h5py.File(path, "r") as f:
                for cam in self.camera_names:
                    img = f["observations/images/{}".format(cam)][t]  # (H, W, 3)
                    img = torch.from_numpy(img.astype(np.float32))
                    if img.max() > 2.0:  # uint8 range
                        img = img / 255.0
                    img = img.permute(2, 0, 1)  # (3, H, W)
                    img = self._img_transform(img)
                    result["vision_{}".format(cam)] = img

                # Load tactile image from HDF5
                if self.use_tactile:
                    tac_img = f["observations/tac/{}/{}".format(
                        self.tac_side, self.tac_key)][t]  # (H, W, 3)
                    tac_img = torch.from_numpy(tac_img.astype(np.float32))
                    if tac_img.max() > 2.0:
                        tac_img = tac_img / 255.0
                    tac_img = tac_img.permute(2, 0, 1)  # (3, H, W)
                    tac_img = self._img_transform(tac_img)
                    result["tactile"] = tac_img

        # Proprio (normalized)
        p = proprio[t].float()
        result["proprio"] = (p - self.proprio_mean) / self.proprio_std

        # Action chunk (normalized)
        action_chunk = self._get_action_chunk(actions, t)
        result["action_chunk"] = (action_chunk - self.action_mean) / self.action_std

        return result

    def resample(self):
        self._index = []
        for eid in self.episode_ids:
            for _ in range(self.samples_per_episode):
                t = random.randint(0, max(self.max_t - 1, 0))
                self._index.append((eid, t))


def collate_dp(batch: list) -> dict:
    """Collate function for DPDataset."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        result[k] = torch.stack([s[k] for s in batch])
    return result
