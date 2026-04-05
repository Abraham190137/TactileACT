"""
Guided Diffusion Policy with CPM (Contact Physical Model).

Paper-faithful TouchGuide implementation (Eq. 11):
    ε̂ = ε_θ - η * sqrt(1 - ᾱ_t) * ∇_A s(V_t, T_t, A_t^k)

Unlike guided_dp.py (which uses TFM + TFS), this version:
  - Uses CPM directly: only needs current (V_t, T_t), no future tactile prediction
  - No TFM needed at all
  - Simpler pipeline: base DP + CPM gradient guidance

Pipeline at inference time:
    1. Encode current observation with DP's vision encoder
    2. Extract DINOv2 features for CPM (current vision + tactile)
    3. Standard DDPM denoising starts from pure noise
    4. At each qualifying denoising step:
       a. CPM scores the current noisy action given (V_t, T_t)
       b. Gradient of CPM score w.r.t. action is computed
       c. Paper Eq. 11: ε̂ = ε_θ - η * sqrt(1 - ᾱ_t) * ∇_A score
    5. Final denoised action is returned
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from tactile_foresight.models.dp_policy import DiffusionPolicy
from tactile_foresight.models.cpm import ContactPhysicalModel


class GuidedDPWithCPM:
    """Wraps a frozen Diffusion Policy with CPM gradient guidance.

    Usage:
        guided = GuidedDPWithCPM(
            dp_policy=dp_model,
            cpm=cpm_model,
            action_stats=action_stats,
        )
        action = guided.predict(obs, cpm_obs)
    """

    def __init__(
        self,
        dp_policy: DiffusionPolicy,
        cpm: ContactPhysicalModel,
        action_stats: Dict[str, torch.Tensor],
        # Diffusion settings
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        # Guidance settings (paper: η)
        guidance_scale: float = 0.1,
        guidance_start_ratio: float = 0.5,  # fraction of denoising steps WITH guidance (from end)
        grad_clip: float = 1.0,
        device: str = "cuda",
    ):
        self.dp_policy = dp_policy
        self.cpm = cpm
        self.action_stats = {
            k: v.to(device) for k, v in action_stats.items()
        }
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_start_ratio = guidance_start_ratio
        self.grad_clip = grad_clip
        self.device = device

        # Build noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Precompute alphas_cumprod for paper Eq. 11
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)

        # Freeze everything
        self.dp_policy.eval()
        self.cpm.eval()
        for p in self.dp_policy.parameters():
            p.requires_grad = False
        for p in self.cpm.parameters():
            p.requires_grad = False

    def _compute_cpm_guidance(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: torch.Tensor,
        noisy_action: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Compute CPM gradient guidance (paper Eq. 11).

        Args:
            vision_feat: (B, 768) DINOv2 vision feature
            tactile_feat: (B, 768) DINOv2 tactile feature
            noisy_action: (B, H, action_dim) current noisy action (normalized)
            t: current denoising timestep

        Returns:
            guidance: (B, H, action_dim) gradient term to subtract from noise pred
        """
        # Compute gradient w.r.t. normalized action directly.
        # Let autograd handle the chain rule through denormalization.
        action_norm = noisy_action.detach().clone().requires_grad_(True)
        action_denorm = (action_norm * self.action_stats["std"]
                         + self.action_stats["mean"])

        # CPM score: s(V_t, T_t, A)
        score = self.cpm.score(vision_feat, tactile_feat, action_denorm)
        total_score = score.sum()

        # Gradient w.r.t. normalized action (autograd handles chain rule)
        grad = torch.autograd.grad(total_score, action_norm)[0]

        # Clip gradient
        if self.grad_clip > 0:
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad = grad * torch.clamp(self.grad_clip / grad_norm, max=1.0)

        # Paper Eq. 11: scale by sqrt(1 - ᾱ_t)
        alpha_t = self.alphas_cumprod[t]
        scale = math.sqrt(1.0 - alpha_t.item())

        return self.guidance_scale * scale * grad

    def predict(
        self,
        dp_obs: Dict[str, torch.Tensor],
        vision_feat: torch.Tensor,
        tactile_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Run guided diffusion policy inference.

        Args:
            dp_obs: observation dict for DP (vision_*, proprio)
            vision_feat: (B, 768) DINOv2 vision feature for CPM
            tactile_feat: (B, 768) DINOv2 tactile feature for CPM

        Returns:
            action: (B, H, action_dim) predicted action chunk (normalized)
        """
        device = self.device
        B = dp_obs["proprio"].shape[0]

        # Encode obs for DP
        with torch.no_grad():
            obs_cond = self.dp_policy.encode_obs(dp_obs)

        # Start from pure noise
        noisy_action = torch.randn(
            (B, self.dp_policy.pred_horizon, self.dp_policy.action_dim),
            device=device,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.noise_scheduler.timesteps

        # Determine which steps get guidance (later portion of denoising)
        total_steps = len(timesteps)
        guidance_start_idx = int(total_steps * (1.0 - self.guidance_start_ratio))

        for step_idx, k in enumerate(timesteps):
            # Base DP noise prediction
            with torch.no_grad():
                noise_pred = self.dp_policy.noise_pred_net(
                    sample=noisy_action,
                    timestep=k,
                    global_cond=obs_cond,
                )

            # Apply CPM guidance in later denoising steps
            apply_guidance = (
                step_idx >= guidance_start_idx
                and self.guidance_scale > 0
            )

            if apply_guidance:
                with torch.enable_grad():
                    guidance = self._compute_cpm_guidance(
                        vision_feat, tactile_feat, noisy_action, k.item()
                    )
                # Paper Eq. 11: ε̂ = ε_θ - guidance
                noise_pred = noise_pred - guidance

            # Standard DDPM step; detach to free intermediate computation graph
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action,
            ).prev_sample.detach()

        return noisy_action

    @torch.no_grad()
    def predict_without_guidance(
        self,
        dp_obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run base DP without guidance (for comparison/ablation)."""
        return self.dp_policy.sample(
            dp_obs, self.noise_scheduler, self.num_inference_steps)


def load_guided_dp_cpm(
    dp_ckpt: str,
    cpm_ckpt: str,
    action_stats_path: str,
    camera_names: list = None,
    guidance_scale: float = 0.1,
    guidance_start_ratio: float = 0.5,
    num_inference_steps: int = 10,
    freeze_vision: bool = True,
    clip_weights_path: Optional[str] = None,
    device: str = "cuda",
) -> GuidedDPWithCPM:
    """Load all components and build CPM-guided DP.

    Args:
        dp_ckpt: path to DP checkpoint (.pth)
        cpm_ckpt: path to CPM checkpoint (.pth)
        action_stats_path: path to action_stats.pt
        camera_names: override camera names (if None, uses checkpoint's)
        guidance_scale: η in paper Eq. 11
        guidance_start_ratio: fraction of denoising with guidance (from end)
        num_inference_steps: DDPM inference steps
        freeze_vision: freeze DP vision backbone
        clip_weights_path: CLIP weights for clip_resnet18 mode
        device: compute device
    """
    if camera_names is None:
        camera_names = ["global", "wrist"]

    # Load DP
    dp_policy = DiffusionPolicy.load_from_checkpoint(
        dp_ckpt, device=device, freeze_vision=freeze_vision,
        clip_weights_path=clip_weights_path,
    )

    # Load CPM — single load, use all saved architecture params
    cpm_ckpt_data = torch.load(cpm_ckpt, map_location="cpu", weights_only=False)
    cpm = ContactPhysicalModel(
        dino_dim=cpm_ckpt_data.get("dino_dim", 768),
        action_dim=cpm_ckpt_data["action_dim"],
        hidden_dim=cpm_ckpt_data["hidden_dim"],
        num_layers=cpm_ckpt_data.get("num_layers", 4),
        nheads=cpm_ckpt_data.get("nheads", 8),
        pred_horizon=cpm_ckpt_data["pred_horizon"],
        dropout=cpm_ckpt_data.get("dropout", 0.1),
        device=device,
    )
    # Load weights from already-loaded checkpoint data
    cpm.obs_encoder.load_state_dict(cpm_ckpt_data["obs_encoder"])
    cpm.action_encoder.load_state_dict(cpm_ckpt_data["action_encoder"])
    cpm.log_temperature.data = cpm_ckpt_data["log_temperature"]

    # Load action stats
    action_stats = torch.load(action_stats_path, map_location="cpu",
                               weights_only=False)

    return GuidedDPWithCPM(
        dp_policy=dp_policy,
        cpm=cpm,
        action_stats=action_stats,
        guidance_scale=guidance_scale,
        guidance_start_ratio=guidance_start_ratio,
        num_inference_steps=num_inference_steps,
        device=device,
    )
