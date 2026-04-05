"""
基于预提取 DINOv2 特征的 TFM v2 数据集。

加载预提取的 768-dim 向量，采样 (V_t, T_t, T_{t+1:t+H}) 用于序列预测 TFM 训练。
"""
from __future__ import annotations

import os
import random
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


class ForesightFeatureDataset(Dataset):
    """加载预提取的 DINOv2 特征，用于 TFM v2 序列预测训练。

    每个 item 返回：
        vision_feat: (768,) float32 — 当前时刻视觉特征（随机选一个摄像头）
        tac_current_feat: (768,) float32 — 当前时刻触觉特征
        tac_future_seq: (H, 768) float32 — 未来 H 步触觉特征序列
        tac_mask: bool — 是否 mask 当前触觉
    """

    def __init__(
        self,
        episode_ids: Sequence[int],
        feature_dir: str,
        camera_names: List[str],
        pred_horizon: int = 20,
        tac_mask_ratio: float = 0.5,
        samples_per_episode: int = 10,
    ):
        self.episode_ids = list(episode_ids)
        self.feature_dir = feature_dir
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.tac_mask_ratio = tac_mask_ratio
        self.samples_per_episode = samples_per_episode

        # 预加载所有特征到内存
        self.all_features = {}
        for eid in episode_ids:
            path = os.path.join(feature_dir, f"episode_{eid}_dino.pt")
            self.all_features[eid] = torch.load(path, map_location="cpu", weights_only=False)

        # 获取轨迹长度
        sample = self.all_features[episode_ids[0]]
        self.episode_length = sample["tactile"].shape[0]

        # 构建 flat index: (episode_id, timestep)
        self._index = []
        self.max_t = self.episode_length - self.pred_horizon  # t+1 到 t+H 不越界
        for eid in self.episode_ids:
            for _ in range(samples_per_episode):
                t = random.randint(0, self.max_t - 1)
                self._index.append((eid, t))

        print(f"[ForesightFeatureDataset] {len(self.episode_ids)} episodes, "
              f"{len(self._index)} samples, pred_horizon={pred_horizon}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        eid, t = self._index[idx]
        feats = self.all_features[eid]

        # 随机选一个摄像头
        cam_name = random.choice(self.camera_names)
        vision_feat = feats[f"vision_{cam_name}"][t].float()  # (768,)

        # 触觉特征
        tac_current_feat = feats["tactile"][t].float()  # (768,)
        # 未来序列: t+1 到 t+H
        tac_future_seq = feats["tactile"][t + 1: t + 1 + self.pred_horizon].float()  # (H, 768)

        # 随机 mask
        tac_mask = random.random() < self.tac_mask_ratio

        return {
            "vision_feat": vision_feat,
            "tac_current_feat": tac_current_feat,
            "tac_future_seq": tac_future_seq,
            "tac_mask": tac_mask,
        }

    def resample(self):
        """每个 epoch 开始时重新采样时间步，增加数据多样性。"""
        self._index = []
        for eid in self.episode_ids:
            for _ in range(self.samples_per_episode):
                t = random.randint(0, self.max_t - 1)
                self._index.append((eid, t))


def collate_foresight_features(batch: list) -> dict:
    """Collate function for ForesightFeatureDataset."""
    return {
        "vision_feat": torch.stack([s["vision_feat"] for s in batch]),
        "tac_current_feat": torch.stack([s["tac_current_feat"] for s in batch]),
        "tac_future_seq": torch.stack([s["tac_future_seq"] for s in batch]),
        "tac_mask": torch.tensor([s["tac_mask"] for s in batch], dtype=torch.bool),
    }
