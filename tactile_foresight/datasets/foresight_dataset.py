"""
ForesightDataset: samples (V_t, T_t, T_{t+h}, qpos_t, action_chunk) tuples
for training TFM and TFS.

HDF5 data layout (xiaomi format):
    observations/images/{cam_name}  (T, H, W, 3) uint8
    observations/tac/{side}/img     (T, 240, 240, 3) uint8
    observations/{proprio_key}      (T, state_dim)
    actions/{action_key} or action  (T, state_dim)
"""
from __future__ import annotations

import os
import random
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ForesightDataset(Dataset):
    """Dataset that yields time-aligned vision-tactile pairs for TFM/TFS.

    Each sample contains:
        - vision_images: dict {cam_name: (3, H, W) float32} current visual obs
        - tac_current: (3, Ht, Wt) float32, current tactile image
        - tac_future: (3, Ht, Wt) float32, future tactile at t+h
        - qpos: (state_dim,) float32, current proprioception
        - action_chunk: (chunk_size, action_dim) float32, action sequence from t
        - horizon: int, the prediction horizon h used
        - tac_mask: bool, whether current tactile is masked (for TFM masking training)
    """

    def __init__(
        self,
        episode_ids: Sequence[int],
        dataset_dir: str,
        camera_names: list[str],
        horizons: list[int] = [4, 8, 12],
        chunk_size: int = 20,
        proprio_key: str = "proprio_eef",
        action_key: str = "actions/joint_abs",
        tac_side: str = "left",
        tac_img_key: str = "img",
        tac_mask_ratio: float = 0.5,
        image_size: tuple[int, int] | None = None,
        tac_image_size: tuple[int, int] | None = (224, 224),
    ):
        self.episode_ids = list(episode_ids)
        self.dataset_dir = dataset_dir
        self.camera_names = [c for c in camera_names if c not in ("gelsight", "blank")]
        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.chunk_size = chunk_size
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.tac_side = tac_side
        self.tac_img_key = tac_img_key
        self.tac_mask_ratio = tac_mask_ratio
        self.image_size = image_size
        self.tac_image_size = tac_image_size

        # ImageNet normalization (same as DINOv2 expects)
        self.img_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Build flat index: (episode_idx, timestep) pairs
        # Only include timesteps where t + max_horizon < episode_len
        # and t + chunk_size <= episode_len (for action chunk)
        self._index = []
        for ep_id in self.episode_ids:
            ep_path = os.path.join(dataset_dir, f"episode_{ep_id}.hdf5")
            with h5py.File(ep_path, "r") as f:
                ep_len = f[f"observations/{self.proprio_key}"].shape[0]
            max_t = ep_len - max(self.max_horizon, self.chunk_size)
            for t in range(max(0, max_t)):
                self._index.append((ep_id, t, ep_len))

        print(f"[ForesightDataset] {len(self.episode_ids)} episodes, "
              f"{len(self._index)} samples, horizons={horizons}, "
              f"chunk_size={chunk_size}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_id, t, ep_len = self._index[idx]

        # Randomly pick a horizon for this sample
        h = random.choice(self.horizons)
        t_future = min(t + h, ep_len - 1)

        ep_path = os.path.join(self.dataset_dir, f"episode_{ep_id}.hdf5")

        with h5py.File(ep_path, "r") as f:
            # --- Visual observations at time t ---
            vision_images = {}
            for cam_name in self.camera_names:
                img = f[f"observations/images/{cam_name}"][t]  # (H, W, 3) uint8
                img = self._process_image(img, self.image_size)
                vision_images[cam_name] = img

            # --- Tactile at time t (current) ---
            tac_path = f"observations/tac/{self.tac_side}/{self.tac_img_key}"
            tac_current = f[tac_path][t]  # (H, W, 3) uint8
            tac_current = self._process_image(tac_current, self.tac_image_size)

            # --- Tactile at time t+h (future target) ---
            tac_future = f[tac_path][t_future]  # (H, W, 3) uint8
            tac_future = self._process_image(tac_future, self.tac_image_size)

            # --- Proprioception at time t ---
            qpos = f[f"observations/{self.proprio_key}"][t].astype(np.float32)

            # --- Action chunk from t ---
            action_end = min(t + self.chunk_size, ep_len)
            action_len = action_end - t
            action = f[f"{self.action_key}"][t:action_end].astype(np.float32)

        # Pad action chunk if needed
        if action_len < self.chunk_size:
            padded = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
            padded[:action_len] = action
            # Repeat last action for padding
            padded[action_len:] = action[-1]
            action = padded

        # Random tactile masking for TFM training
        tac_mask = random.random() < self.tac_mask_ratio

        return {
            "vision_images": vision_images,       # dict of (3, H, W)
            "tac_current": tac_current,            # (3, Ht, Wt)
            "tac_future": tac_future,              # (3, Ht, Wt)
            "qpos": torch.from_numpy(qpos),        # (state_dim,)
            "action_chunk": torch.from_numpy(action),  # (chunk_size, action_dim)
            "horizon": h,
            "tac_mask": tac_mask,
        }

    def _process_image(
        self, img: np.ndarray, target_size: tuple[int, int] | None
    ) -> torch.Tensor:
        """Convert uint8 HWC image to normalized CHW float tensor."""
        img = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        if target_size is not None:
            h, w = target_size
            if t.shape[1] != h or t.shape[2] != w:
                t = T.functional.resize(
                    t, [h, w],
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                )

        t = self.img_normalize(t)
        return t


def collate_foresight(batch: list[dict]) -> dict:
    """Custom collate that stacks vision images per camera."""
    keys_simple = ["tac_current", "tac_future", "qpos", "action_chunk"]
    out = {}
    for k in keys_simple:
        out[k] = torch.stack([s[k] for s in batch])

    # Stack vision images per camera
    cam_names = list(batch[0]["vision_images"].keys())
    out["vision_images"] = {
        cam: torch.stack([s["vision_images"][cam] for s in batch])
        for cam in cam_names
    }

    out["horizon"] = torch.tensor([s["horizon"] for s in batch], dtype=torch.long)
    out["tac_mask"] = torch.tensor([s["tac_mask"] for s in batch], dtype=torch.bool)

    return out


if __name__ == "__main__":
    dataset_dir = "/home/chenshuai/data/dataset/260309_0310"
    ds = ForesightDataset(
        episode_ids=list(range(10)),
        dataset_dir=dataset_dir,
        camera_names=["global", "wrist"],
        horizons=[4, 8, 12],
        chunk_size=20,
    )
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k}/{kk}: {vv.shape}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")
