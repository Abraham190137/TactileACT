"""
Vision-Tactile Alignment with DINOv2 + Projection Heads.

Aligns visual and tactile representations in a shared embedding space
using CLIP-style contrastive learning. DINOv2 ViT-B/14 is frozen as
the backbone; only lightweight projection heads are trained.

Architecture:
    Vision image  → DINOv2 frozen (768) → VisionProjHead (768→512→256) → L2 norm
    Tactile image → DINOv2 frozen (768) → TactileProjHead (768→512→256) → L2 norm
    Loss: InfoNCE / CLIP contrastive loss in shared 256-dim space

After training, the projection heads map DINOv2 features into a shared
space where vision and tactile embeddings of the same timestep are close.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import DINOv2Extractor


class ProjectionHead(nn.Module):
    """MLP projection head: DINOv2 dim → shared dim with L2 normalization."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 256,
        conditioning_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + conditioning_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 768) DINOv2 embedding
            conditioning: (B, conditioning_dim) optional extra features (e.g. proprio)
        Returns:
            (B, out_dim) L2-normalized embedding
        """
        if conditioning is not None:
            x = torch.cat([x, conditioning], dim=-1)
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x


class VTAlignmentModel(nn.Module):
    """Vision-Tactile Alignment model.

    Uses frozen DINOv2 + trainable projection heads to align
    vision and tactile representations in a shared embedding space.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        hidden_dim: int = 512,
        conditioning_dim: int = 0,
        dropout: float = 0.1,
        learnable_temperature: bool = True,
        init_temperature: float = 0.07,
        max_temperature: float = 100.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.max_log_temperature = np.log(max_temperature)

        # Frozen DINOv2 backbone (shared for vision and tactile)
        self.dino = DINOv2Extractor(device=device)
        dino_dim = self.dino.embed_dim  # 768

        # Trainable projection heads
        self.vision_proj = ProjectionHead(
            in_dim=dino_dim, hidden_dim=hidden_dim, out_dim=shared_dim,
            dropout=dropout,
        )
        self.tactile_proj = ProjectionHead(
            in_dim=dino_dim, hidden_dim=hidden_dim, out_dim=shared_dim,
            conditioning_dim=conditioning_dim, dropout=dropout,
        )

        # Learnable temperature for contrastive loss
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(np.log(1.0 / init_temperature))
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(np.log(1.0 / init_temperature)),
            )

        self.to(device)

    @property
    def temperature(self) -> torch.Tensor:
        # clamp 防止温度失控
        clamped = torch.clamp(self.log_temperature, max=self.max_log_temperature)
        return clamped.exp()

    def encode_vision(
        self, images: torch.Tensor, return_dino: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Encode vision images to shared space.

        Args:
            images: (B, 3, H, W) already ImageNet-normalized
            return_dino: if True, also return raw DINOv2 features

        Returns:
            (B, shared_dim) projected embedding, or tuple with DINOv2 features
        """
        with torch.no_grad():
            dino_feat = self.dino(images, already_normalized=True)  # (B, 768)
        proj = self.vision_proj(dino_feat)  # (B, shared_dim)
        if return_dino:
            return proj, dino_feat
        return proj

    def encode_tactile(
        self,
        images: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        return_dino: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Encode tactile images to shared space.

        Args:
            images: (B, 3, H, W) already ImageNet-normalized
            conditioning: (B, D) optional proprio conditioning
            return_dino: if True, also return raw DINOv2 features

        Returns:
            (B, shared_dim) projected embedding, or tuple with DINOv2 features
        """
        with torch.no_grad():
            dino_feat = self.dino(images, already_normalized=True)  # (B, 768)
        proj = self.tactile_proj(dino_feat, conditioning)  # (B, shared_dim)
        if return_dino:
            return proj, dino_feat
        return proj

    def compute_loss(
        self,
        vision_imgs: torch.Tensor,
        tactile_imgs: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute CLIP-style contrastive loss.

        Args:
            vision_imgs: (B, N, 3, H, W) — N timesteps per episode
            tactile_imgs: (B, N, 3, Ht, Wt)
            conditioning: (B, N, D) optional proprio

        Returns:
            dict with 'loss', 'accuracy', 'temperature'
        """
        B, N = vision_imgs.shape[:2]
        device = vision_imgs.device

        # Flatten batch and timestep dims
        v_flat = vision_imgs.reshape(B * N, *vision_imgs.shape[2:])
        t_flat = tactile_imgs.reshape(B * N, *tactile_imgs.shape[2:])

        with torch.no_grad():
            v_dino = self.dino(v_flat, already_normalized=True)  # (B*N, 768)
            t_dino = self.dino(t_flat, already_normalized=True)  # (B*N, 768)

        v_emb = self.vision_proj(v_dino)    # (B*N, shared_dim)
        if conditioning is not None:
            c_flat = conditioning.reshape(B * N, -1)
            t_emb = self.tactile_proj(t_dino, c_flat)
        else:
            t_emb = self.tactile_proj(t_dino)  # (B*N, shared_dim)

        # Reshape back: (B, N, shared_dim)
        v_emb = v_emb.view(B, N, self.shared_dim)
        t_emb = t_emb.view(B, N, self.shared_dim)

        # Per-batch contrastive loss: within each batch item,
        # match N vision-tactile pairs
        temp = self.temperature
        target = torch.eye(N, device=device)  # (N, N)

        total_loss = 0.0
        correct = 0
        total = 0

        for b in range(B):
            # (N, shared_dim) @ (shared_dim, N) → (N, N)
            logits_v2t = temp * v_emb[b] @ t_emb[b].T
            logits_t2v = temp * t_emb[b] @ v_emb[b].T

            loss_v2t = F.cross_entropy(logits_v2t, target)
            loss_t2v = F.cross_entropy(logits_t2v, target)
            total_loss += (loss_v2t + loss_t2v) / 2.0

            # accuracy: fraction of correct top-1 matches
            with torch.no_grad():
                pred_v = logits_v2t.argmax(dim=1)
                pred_t = logits_t2v.argmax(dim=1)
                labels = torch.arange(N, device=device)
                correct += (pred_v == labels).sum().item()
                correct += (pred_t == labels).sum().item()
                total += 2 * N

        loss = total_loss / B
        accuracy = correct / max(total, 1)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "temperature": temp.item(),
        }

    def compute_loss_multicam(
        self,
        vision_imgs: torch.Tensor,
        tactile_imgs: torch.Tensor,
    ) -> dict:
        """多摄像头对比 loss，触觉 DINOv2 只提取一次。

        Args:
            vision_imgs: (B, N, Cams, 3, H, W) — 多摄像头
            tactile_imgs: (B, N, 3, Ht, Wt)

        Returns:
            dict with 'loss', 'accuracy', 'temperature'
        """
        B, N, Cams = vision_imgs.shape[:3]
        device = vision_imgs.device

        # 触觉 DINOv2 只提一次
        t_flat = tactile_imgs.reshape(B * N, *tactile_imgs.shape[2:])
        with torch.no_grad():
            t_dino = self.dino(t_flat, already_normalized=True)  # (B*N, 768)
        t_emb = self.tactile_proj(t_dino).view(B, N, self.shared_dim)  # (B, N, D)

        temp = self.temperature
        target = torch.eye(N, device=device)

        total_loss = 0.0
        correct = 0
        total = 0

        for cam_idx in range(Cams):
            cam_flat = vision_imgs[:, :, cam_idx].reshape(B * N, *vision_imgs.shape[3:])
            with torch.no_grad():
                v_dino = self.dino(cam_flat, already_normalized=True)
            v_emb = self.vision_proj(v_dino).view(B, N, self.shared_dim)

            for b in range(B):
                logits_v2t = temp * v_emb[b] @ t_emb[b].T
                logits_t2v = temp * t_emb[b] @ v_emb[b].T

                loss_v2t = F.cross_entropy(logits_v2t, target)
                loss_t2v = F.cross_entropy(logits_t2v, target)
                total_loss += (loss_v2t + loss_t2v) / 2.0

                with torch.no_grad():
                    pred_v = logits_v2t.argmax(dim=1)
                    pred_t = logits_t2v.argmax(dim=1)
                    labels = torch.arange(N, device=device)
                    correct += (pred_v == labels).sum().item()
                    correct += (pred_t == labels).sum().item()
                    total += 2 * N

        loss = total_loss / (B * Cams)
        accuracy = correct / max(total, 1)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "temperature": temp.item(),
        }

    def compute_loss_from_features(
        self,
        vision_feats: torch.Tensor,
        tactile_feats: torch.Tensor,
    ) -> dict:
        """从预提取的 DINOv2 特征直接计算多摄像头对比 loss，跳过 DINOv2 前向。

        Args:
            vision_feats: (B, N, Cams, 768) 预提取的视觉 DINOv2 特征
            tactile_feats: (B, N, 768) 预提取的触觉 DINOv2 特征

        Returns:
            dict with 'loss', 'accuracy', 'temperature'
        """
        B, N, Cams, D = vision_feats.shape
        device = vision_feats.device

        # 触觉 projection: (B*N, 768) → (B, N, shared_dim)
        t_flat = tactile_feats.reshape(B * N, D)
        t_emb = self.tactile_proj(t_flat).view(B, N, self.shared_dim)

        temp = self.temperature
        target = torch.eye(N, device=device)

        total_loss = 0.0
        correct = 0
        total = 0

        for cam_idx in range(Cams):
            v_flat = vision_feats[:, :, cam_idx].reshape(B * N, D)
            v_emb = self.vision_proj(v_flat).view(B, N, self.shared_dim)

            for b in range(B):
                logits_v2t = temp * v_emb[b] @ t_emb[b].T
                logits_t2v = temp * t_emb[b] @ v_emb[b].T

                loss_v2t = F.cross_entropy(logits_v2t, target)
                loss_t2v = F.cross_entropy(logits_t2v, target)
                total_loss += (loss_v2t + loss_t2v) / 2.0

                with torch.no_grad():
                    pred_v = logits_v2t.argmax(dim=1)
                    pred_t = logits_t2v.argmax(dim=1)
                    labels = torch.arange(N, device=device)
                    correct += (pred_v == labels).sum().item()
                    correct += (pred_t == labels).sum().item()
                    total += 2 * N

        loss = total_loss / (B * Cams)
        accuracy = correct / max(total, 1)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "temperature": temp.item(),
        }

    def trainable_parameters(self):
        """Return only trainable parameters (projection heads + temperature)."""
        params = list(self.vision_proj.parameters()) + list(self.tactile_proj.parameters())
        if self.log_temperature.requires_grad:
            params.append(self.log_temperature)
        return params

    def num_trainable_params(self) -> int:
        return sum(
            p.numel() for p in self.trainable_parameters() if p.requires_grad
        )

    def save_heads(self, path: str):
        """Save only the projection heads and temperature."""
        torch.save({
            "vision_proj": self.vision_proj.state_dict(),
            "tactile_proj": self.tactile_proj.state_dict(),
            "log_temperature": self.log_temperature.data,
            "shared_dim": self.shared_dim,
        }, path)

    def load_heads(self, path: str):
        """Load projection heads and temperature."""
        ckpt = torch.load(path, map_location="cpu")
        self.vision_proj.load_state_dict(ckpt["vision_proj"])
        self.tactile_proj.load_state_dict(ckpt["tactile_proj"])
        self.log_temperature.data = ckpt["log_temperature"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building VTAlignmentModel on {device}...")

    model = VTAlignmentModel(shared_dim=256, device=device)
    print(f"Trainable params: {model.num_trainable_params():,}")
    print(f"Temperature: {model.temperature.item():.4f}")

    B, N = 2, 5
    vis = torch.randn(B, N, 3, 200, 266, device=device)
    tac = torch.randn(B, N, 3, 224, 224, device=device)

    result = model.compute_loss(vis, tac)
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("VTAlignment OK!")
