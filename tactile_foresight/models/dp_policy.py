"""
Diffusion Policy (DP) — standalone implementation for tactile_foresight.

Supports 4 vision encoder modes:
  - dino_feature:  precomputed DINOv2 768-dim features -> MLP(768->512)
  - dino_online:   frozen DINOv2 ViT-B/14 -> MLP(768->512), for training+inference
  - resnet18:      ResNet18(BN->GN), random init, end-to-end trainable
  - clip_resnet18: CLIP-pretrained ResNet18(GN), load weights from --clip_weights_path

Tactile encoder (optional):
  - clip_resnet18 mode: separate CLIP-pretrained ResNet18 for tactile images
  - Loaded from --clip_tac_weights_path, can be frozen with --freeze_tactile

Architecture:
  obs_cond = concat(per_camera_features, [tactile_feature], proprio) -> ConditionalUnet1D
  Training: DDPM noise prediction (epsilon)
  Inference: DDPM iterative denoising

Reuses diffusion/network.py ConditionalUnet1D without modification.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# Add project root so we can import from diffusion/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusion.network import ConditionalUnet1D, get_resnet, replace_bn_with_gn


# ---------------------------------------------------------------------------
# Vision Encoders
# ---------------------------------------------------------------------------

class DinoFeatureEncoder(nn.Module):
    """MLP projection from precomputed DINOv2 768-dim to 512-dim."""

    def __init__(self, dino_dim=768, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dino_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, feat):
        """feat: (B, 768) precomputed DINOv2 feature."""
        return self.proj(feat)


class DinoOnlineEncoder(nn.Module):
    """Frozen DINOv2 backbone + trainable MLP projection."""

    def __init__(self, out_dim=512, freeze_backbone=True):
        super().__init__()
        from tactile_foresight.models.feature_extractor import (
            DINOv2Extractor, _patch_dinov2_for_py38,
        )
        self.extractor = DINOv2Extractor(device="cpu")
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.extractor.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, images, already_normalized=True):
        """images: (B, 3, H, W) float tensor."""
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.extractor(images, already_normalized=already_normalized)
        else:
            feat = self.extractor(images, already_normalized=already_normalized)
        return self.proj(feat)


class ResNet18Encoder(nn.Module):
    """ResNet18 with BN->GN, random init. Output 512-dim."""

    def __init__(self, freeze=False):
        super().__init__()
        self.encoder = get_resnet("resnet18")
        self.encoder = replace_bn_with_gn(self.encoder)
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, images):
        """images: (B, 3, H, W) float tensor."""
        if self.freeze:
            with torch.no_grad():
                return self.encoder(images)
        return self.encoder(images)


class ClipResNet18Encoder(nn.Module):
    """CLIP-pretrained ResNet18(GN) + AdaptiveAvgPool + Flatten -> 512-dim.

    Uses modified_resnet18 architecture (no avgpool/fc, BN->GN) matching
    clip_pretraining_xiaomi.py, then loads pretrained weights.
    """

    def __init__(self, clip_weights_path=None, freeze=False):
        super().__init__()
        # Build modified_resnet18: ResNet18 without avgpool+fc, BN->GN
        resnet18 = getattr(torchvision.models, "resnet18")()
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])
        self.backbone = replace_bn_with_gn(self.backbone)

        if clip_weights_path is not None and os.path.exists(clip_weights_path):
            state_dict = torch.load(clip_weights_path, map_location="cpu",
                                    weights_only=False)
            self.backbone.load_state_dict(state_dict, strict=False)
            print("[ClipResNet18Encoder] Loaded CLIP weights from {}".format(
                clip_weights_path))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1, -1)

        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        """images: (B, 3, H, W) float tensor -> (B, 512)."""
        if self.freeze:
            with torch.no_grad():
                feat_map = self.backbone(images)
        else:
            feat_map = self.backbone(images)
        return self.flatten(self.pool(feat_map))


# ---------------------------------------------------------------------------
# Diffusion Policy
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """Diffusion Policy wrapping ConditionalUnet1D + vision encoder + optional tactile encoder.

    Args:
        obs_encoder_type: one of "dino_feature", "dino_online", "resnet18", "clip_resnet18"
        camera_names: list of camera names (each gets its own encoder for resnet/clip modes)
        action_dim: dimension of action space
        pred_horizon: length of action chunk to predict
        proprio_dim: dimension of proprioceptive observation
        freeze_vision: whether to freeze vision backbone
        clip_weights_path: path to CLIP pretrained weights for vision (for clip_resnet18 mode)
        use_tactile: whether to include tactile encoder
        clip_tac_weights_path: path to CLIP pretrained weights for tactile encoder
        freeze_tactile: whether to freeze tactile backbone
    """

    VISION_FEAT_DIM = {
        "dino_feature": 512,  # after MLP projection
        "dino_online": 512,
        "resnet18": 512,
        "clip_resnet18": 512,
    }

    def __init__(
        self,
        obs_encoder_type: str = "dino_feature",
        camera_names: List[str] = None,
        action_dim: int = 6,
        pred_horizon: int = 20,
        proprio_dim: int = 6,
        freeze_vision: bool = True,
        clip_weights_path: Optional[str] = None,
        use_tactile: bool = False,
        clip_tac_weights_path: Optional[str] = None,
        freeze_tactile: bool = True,
        share_vision_backbone: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        if camera_names is None:
            camera_names = ["global", "wrist"]

        self.obs_encoder_type = obs_encoder_type
        self.camera_names = camera_names
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.proprio_dim = proprio_dim
        self.use_tactile = use_tactile
        self.share_vision_backbone = share_vision_backbone

        # Build vision encoders
        feat_dim_per_cam = self.VISION_FEAT_DIM[obs_encoder_type]

        if share_vision_backbone:
            # ACT-style: one shared backbone for all cameras
            if obs_encoder_type == "dino_feature":
                self.shared_vision_encoder = DinoFeatureEncoder(768, feat_dim_per_cam)
            elif obs_encoder_type == "dino_online":
                self.shared_vision_encoder = DinoOnlineEncoder(
                    out_dim=feat_dim_per_cam, freeze_backbone=freeze_vision)
            elif obs_encoder_type == "resnet18":
                self.shared_vision_encoder = ResNet18Encoder(freeze=freeze_vision)
            elif obs_encoder_type == "clip_resnet18":
                self.shared_vision_encoder = ClipResNet18Encoder(
                    clip_weights_path=clip_weights_path, freeze=freeze_vision)
            else:
                raise ValueError("Unknown obs_encoder_type: {}".format(obs_encoder_type))
            self.vision_encoders = None
            print("[DiffusionPolicy] Shared vision backbone for {} cameras".format(
                len(camera_names)))
        else:
            # Per-camera encoders (original DP style)
            self.shared_vision_encoder = None
            self.vision_encoders = nn.ModuleDict()
            if obs_encoder_type == "dino_feature":
                for cam in camera_names:
                    self.vision_encoders[cam] = DinoFeatureEncoder(768, feat_dim_per_cam)
            elif obs_encoder_type == "dino_online":
                for cam in camera_names:
                    self.vision_encoders[cam] = DinoOnlineEncoder(
                        out_dim=feat_dim_per_cam, freeze_backbone=freeze_vision)
            elif obs_encoder_type == "resnet18":
                for cam in camera_names:
                    self.vision_encoders[cam] = ResNet18Encoder(freeze=freeze_vision)
            elif obs_encoder_type == "clip_resnet18":
                for cam in camera_names:
                    self.vision_encoders[cam] = ClipResNet18Encoder(
                        clip_weights_path=clip_weights_path, freeze=freeze_vision)
            else:
                raise ValueError("Unknown obs_encoder_type: {}".format(obs_encoder_type))

        # Build tactile encoder (optional, separate from vision)
        tactile_feat_dim = 0
        self.tactile_encoder = None
        if use_tactile:
            tactile_feat_dim = 512
            if obs_encoder_type in ("clip_resnet18", "resnet18"):
                self.tactile_encoder = ClipResNet18Encoder(
                    clip_weights_path=clip_tac_weights_path, freeze=freeze_tactile)
            elif obs_encoder_type in ("dino_feature", "dino_online"):
                # For dino modes, tactile uses same architecture as vision
                if obs_encoder_type == "dino_feature":
                    self.tactile_encoder = DinoFeatureEncoder(768, tactile_feat_dim)
                else:
                    self.tactile_encoder = DinoOnlineEncoder(
                        out_dim=tactile_feat_dim, freeze_backbone=freeze_vision)
            print("[DiffusionPolicy] Tactile encoder enabled, feat_dim={}".format(
                tactile_feat_dim))

        # global_cond_dim = num_cameras * feat_dim + [tactile_feat_dim] + proprio_dim
        global_cond_dim = len(camera_names) * feat_dim_per_cam + tactile_feat_dim + proprio_dim

        # Noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
        )

        self.to(device)

    def encode_obs(
        self, obs: Dict[str, torch.Tensor], already_normalized: bool = True,
    ) -> torch.Tensor:
        """Encode observation into conditioning vector.

        Args:
            obs: dict with keys:
              - "vision_{cam}": (B, 768) for dino_feature, or (B, 3, H, W) for image modes
              - "tactile": (B, 768) or (B, 3, H, W) tactile input (if use_tactile)
              - "proprio": (B, proprio_dim) normalized proprioceptive state
            already_normalized: whether images are already ImageNet-normalized
                (only relevant for dino_online mode)

        Returns:
            (B, global_cond_dim) conditioning vector
        """
        features = []
        for cam in self.camera_names:
            key = "vision_{}".format(cam)
            cam_input = obs[key]
            if self.share_vision_backbone:
                encoder = self.shared_vision_encoder
            else:
                encoder = self.vision_encoders[cam]

            if self.obs_encoder_type == "dino_online":
                feat = encoder(cam_input, already_normalized=already_normalized)
            else:
                feat = encoder(cam_input)  # (B, 512)
            features.append(feat)

        # Tactile encoder
        if self.use_tactile and self.tactile_encoder is not None:
            tac_input = obs["tactile"]
            if self.obs_encoder_type == "dino_online":
                tac_feat = self.tactile_encoder(
                    tac_input, already_normalized=already_normalized)
            else:
                tac_feat = self.tactile_encoder(tac_input)  # (B, 512)
            features.append(tac_feat)

        # Concat camera features + [tactile feature] + proprio
        features.append(obs["proprio"])  # (B, proprio_dim)
        return torch.cat(features, dim=-1)  # (B, global_cond_dim)

    def compute_loss(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        noise_scheduler,
    ) -> torch.Tensor:
        """Compute DDPM training loss (MSE on noise prediction).

        Args:
            obs: observation dict
            action: (B, H, action_dim) normalized action chunk
            noise_scheduler: DDPMScheduler instance

        Returns:
            scalar MSE loss
        """
        B = action.shape[0]
        device = action.device

        obs_cond = self.encode_obs(obs)  # (B, global_cond_dim)

        # Sample noise
        noise = torch.randn_like(action)

        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device,
        ).long()

        # Forward diffusion: add noise
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

        # Predict noise
        noise_pred = self.noise_pred_net(
            noisy_action, timesteps, global_cond=obs_cond)

        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(
        self,
        obs: Dict[str, torch.Tensor],
        noise_scheduler,
        num_inference_steps: int = 10,
    ) -> torch.Tensor:
        """Generate action chunk via DDPM denoising.

        Args:
            obs: observation dict
            noise_scheduler: DDPMScheduler instance (will set timesteps internally)
            num_inference_steps: number of denoising steps

        Returns:
            (B, H, action_dim) denoised action chunk (normalized)
        """
        device = next(self.parameters()).device
        # Infer batch size from proprio
        B = obs["proprio"].shape[0]

        obs_cond = self.encode_obs(obs)

        # Start from pure noise
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=device)

        noise_scheduler.set_timesteps(num_inference_steps)
        for t in noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                noisy_action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond,
            )
            noisy_action = noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_action,
            ).prev_sample

        return noisy_action

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "obs_encoder_type": self.obs_encoder_type,
            "camera_names": self.camera_names,
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "proprio_dim": self.proprio_dim,
            "use_tactile": self.use_tactile,
            "share_vision_backbone": self.share_vision_backbone,
        }, path)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: str = "cpu",
        freeze_vision: bool = True,
        clip_weights_path: Optional[str] = None,
        clip_tac_weights_path: Optional[str] = None,
        freeze_tactile: bool = True,
        obs_encoder_override: Optional[str] = None,
    ):
        """Load model from checkpoint.

        Args:
            path: checkpoint path
            device: target device
            freeze_vision: freeze vision backbone
            clip_weights_path: CLIP weights for clip_resnet18 mode (vision)
            clip_tac_weights_path: CLIP weights for tactile encoder
            freeze_tactile: freeze tactile backbone
            obs_encoder_override: override obs_encoder_type from checkpoint.
                Useful for cross-mode loading, e.g. train with "dino_feature"
                then deploy with "dino_online". Compatible pairs:
                  dino_feature -> dino_online (MLP proj weights transfer,
                      DINOv2 backbone loaded from pretrained)
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved_type = ckpt["obs_encoder_type"]
        target_type = obs_encoder_override or saved_type

        model = cls(
            obs_encoder_type=target_type,
            camera_names=ckpt["camera_names"],
            action_dim=ckpt["action_dim"],
            pred_horizon=ckpt["pred_horizon"],
            proprio_dim=ckpt["proprio_dim"],
            freeze_vision=freeze_vision,
            clip_weights_path=clip_weights_path,
            use_tactile=ckpt.get("use_tactile", False),
            clip_tac_weights_path=clip_tac_weights_path,
            freeze_tactile=freeze_tactile,
            share_vision_backbone=ckpt.get("share_vision_backbone", False),
            device=device,
        )

        if target_type == saved_type:
            # Same mode: strict load
            model.load_state_dict(ckpt["state_dict"])
        else:
            # Cross-mode: load matching keys, skip missing (e.g. DINOv2 backbone)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            print("[DiffusionPolicy] Cross-mode load: {} -> {}. "
                  "Loaded matching weights, skipped encoder-specific keys.".format(
                      saved_type, target_type))

        return model


if __name__ == "__main__":
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 4

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # Test dino_feature mode
    print("=== Testing dino_feature mode ===")
    policy = DiffusionPolicy(
        obs_encoder_type="dino_feature",
        camera_names=["global", "wrist"],
        action_dim=6,
        pred_horizon=20,
        proprio_dim=6,
        device=device,
    )
    print("Trainable params: {:,}".format(policy.num_trainable_params()))

    obs = {
        "vision_global": torch.randn(B, 768, device=device),
        "vision_wrist": torch.randn(B, 768, device=device),
        "proprio": torch.randn(B, 6, device=device),
    }
    action = torch.randn(B, 20, 6, device=device)

    loss = policy.compute_loss(obs, action, noise_scheduler)
    print("Loss: {:.4f}".format(loss.item()))

    sampled = policy.sample(obs, noise_scheduler, num_inference_steps=10)
    print("Sampled shape: {}".format(sampled.shape))

    # Test resnet18 mode
    print("\n=== Testing resnet18 mode ===")
    policy2 = DiffusionPolicy(
        obs_encoder_type="resnet18",
        camera_names=["global", "wrist"],
        action_dim=6,
        pred_horizon=20,
        proprio_dim=6,
        freeze_vision=False,
        device=device,
    )
    print("Trainable params: {:,}".format(policy2.num_trainable_params()))

    obs2 = {
        "vision_global": torch.randn(B, 3, 224, 224, device=device),
        "vision_wrist": torch.randn(B, 3, 224, 224, device=device),
        "proprio": torch.randn(B, 6, device=device),
    }
    loss2 = policy2.compute_loss(obs2, action, noise_scheduler)
    print("Loss: {:.4f}".format(loss2.item()))

    print("\nDP Policy OK!")
