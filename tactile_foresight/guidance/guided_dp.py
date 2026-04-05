"""
Guided Diffusion Policy with Tactile Foresight.

During DDPM denoising, injects gradient-based guidance from the TFS module
to steer action generation toward tactile-feasible outcomes.

Pipeline at inference time:
    1. TFM predicts future tactile embedding from current vision (+ optional tactile)
    2. Standard DDPM denoising starts from pure noise
    3. At each qualifying denoising step:
       a. TFS scores the current noisy action given (vision_emb, t̂_emb)
       b. Gradient of TFS score w.r.t. action is computed
       c. Gradient is injected into the denoising step as guidance
    4. Final denoised action is returned

All three modules (base DP, TFM, TFS) are frozen at inference time.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from tactile_foresight.models.tfm import TactileForesightModel
from tactile_foresight.models.tfs import TactileFeasibilityScore


class GuidedDiffusionPolicy:
    """Wraps a frozen Diffusion Policy with TFM + TFS gradient guidance.

    Usage:
        guided = GuidedDiffusionPolicy(
            nets=dp_nets,           # ModuleDict with noise_pred_net + encoders
            tfm=tfm_model,          # frozen TFM
            tfs=tfs_model,          # frozen TFS
            camera_names=["global", "wrist"],
            pred_horizon=20,
            action_dim=7,
        )
        action = guided.predict(image_data, qpos_data)
    """

    def __init__(
        self,
        nets: nn.ModuleDict,
        tfm: TactileForesightModel,
        tfs: TactileFeasibilityScore,
        camera_names: list,
        pred_horizon: int = 20,
        action_dim: int = 7,
        # Diffusion settings
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        # Guidance settings
        guidance_scale: float = 0.1,
        guidance_start_ratio: float = 0.5,
        grad_clip: float = 1.0,
        # TFM settings
        horizon: int = 8,
        device: str = "cuda",
    ):
        self.nets = nets
        self.tfm = tfm
        self.tfs = tfs
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_start_ratio = guidance_start_ratio
        self.grad_clip = grad_clip
        self.horizon = horizon
        self.device = device

        # Build noise scheduler (same config as DP training)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Freeze everything
        self.nets.eval()
        self.tfm.eval()
        self.tfs.eval()
        for p in self.nets.parameters():
            p.requires_grad = False
        for p in self.tfm.parameters():
            p.requires_grad = False
        for p in self.tfs.parameters():
            p.requires_grad = False

    def _encode_obs(self, image_data, qpos_data):
        """Encode observations using the base DP's vision encoders.

        Args:
            image_data: list of (1, obs_horizon, 3, H, W) tensors per camera
            qpos_data: (1, obs_horizon, qpos_dim) or (1, qpos_dim)

        Returns:
            obs_cond: (1, obs_dim) global conditioning for UNet
        """
        with torch.no_grad():
            image_features = torch.Tensor().to(self.device)
            for i, cam_name in enumerate(self.camera_names):
                img = image_data[i].to(self.device)
                if cam_name == "gelsight" and "gelsight_encoder" in self.nets:
                    feat = self.nets["gelsight_encoder"](img.flatten(end_dim=1))
                    feat = feat.reshape(*img.shape[:2], -1)
                elif f"{cam_name}_encoder" in self.nets:
                    feat = self.nets[f"{cam_name}_encoder"](img.flatten(end_dim=1))
                    feat = feat.reshape(*img.shape[:2], -1)
                else:
                    continue
                image_features = torch.cat([image_features, feat], dim=-1)

            agent_pos = qpos_data.to(self.device)
            if agent_pos.dim() == 2:
                agent_pos = agent_pos.unsqueeze(1)
            obs = torch.cat([image_features, agent_pos], dim=-1)
            obs_cond = obs.flatten(start_dim=1)

        return obs_cond

    def _get_tfm_embeddings(self, vision_imgs, tac_current_imgs=None):
        """Get TFM predicted tactile embedding + vision embedding in aligned space.

        Args:
            vision_imgs: (B, 3, H, W) vision images for TFM
            tac_current_imgs: (B, 3, H, W) optional current tactile images

        Returns:
            v_emb: (B, aligned_dim) aligned vision embedding
            t_hat: (B, aligned_dim) predicted future tactile embedding
        """
        with torch.no_grad():
            v_emb = self.tfm.encode_vision(vision_imgs)

        horizon_tensor = torch.tensor(
            [self.horizon], device=self.device
        ).expand(vision_imgs.shape[0])

        t_hat = self.tfm.predict(
            vision_imgs, tac_current_imgs, horizon_tensor
        )

        return v_emb, t_hat.detach()

    def _compute_guidance(self, v_emb, t_hat, noisy_action):
        """Compute TFS gradient guidance.

        Args:
            v_emb: (B, aligned_dim) frozen vision embedding
            t_hat: (B, aligned_dim) frozen predicted tactile embedding
            noisy_action: (B, T, D) current noisy action (will be detached + requires_grad)

        Returns:
            guidance_grad: (B, T, D) gradient to subtract from noise prediction
        """
        action = noisy_action.detach().requires_grad_(True)

        # TFS score
        score = self.tfs.score(v_emb, t_hat, action)
        total_score = score.sum()

        # Compute gradient
        grad = torch.autograd.grad(total_score, action)[0]

        # Clip gradient magnitude
        if self.grad_clip > 0:
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad = grad * torch.clamp(self.grad_clip / grad_norm, max=1.0)

        return grad

    @torch.no_grad()
    def predict(
        self,
        image_data: list,
        qpos_data: torch.Tensor,
        vision_imgs_for_tfm: Optional[torch.Tensor] = None,
        tac_current_imgs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run guided diffusion policy inference.

        Args:
            image_data: list of tensors per camera (for base DP encoder)
            qpos_data: proprioceptive state
            vision_imgs_for_tfm: (B, 3, H, W) vision images for TFM input
                (separate from DP encoder input; should be ImageNet-normalized)
            tac_current_imgs: (B, 3, H, W) optional current tactile for TFM

        Returns:
            action: (B, pred_horizon, action_dim) predicted action chunk
        """
        # 1. Encode observations for base DP
        obs_cond = self._encode_obs(image_data, qpos_data)
        B = obs_cond.shape[0]

        # 2. Get TFM predictions (if vision images provided)
        v_emb, t_hat = None, None
        if vision_imgs_for_tfm is not None:
            v_emb, t_hat = self._get_tfm_embeddings(
                vision_imgs_for_tfm, tac_current_imgs
            )

        # 3. DDPM denoising with guidance
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device
        )
        naction = noisy_action

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.noise_scheduler.timesteps

        # Determine which steps get guidance
        total_steps = len(timesteps)
        guidance_start_idx = int(total_steps * (1.0 - self.guidance_start_ratio))

        for step_idx, k in enumerate(timesteps):
            # Base DP noise prediction
            noise_pred = self.nets["noise_pred_net"](
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )

            # Apply guidance in the later portion of denoising
            apply_guidance = (
                v_emb is not None
                and t_hat is not None
                and step_idx >= guidance_start_idx
                and self.guidance_scale > 0
            )

            if apply_guidance:
                # Need gradients for this computation
                with torch.enable_grad():
                    grad = self._compute_guidance(v_emb, t_hat, naction)
                # Guidance: subtract scaled gradient from noise prediction
                # (minimizing noise in the direction of higher TFS score)
                noise_pred = noise_pred - self.guidance_scale * grad

            # Standard DDPM step
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        return naction.detach()

    @torch.no_grad()
    def predict_without_guidance(
        self,
        image_data: list,
        qpos_data: torch.Tensor,
    ) -> torch.Tensor:
        """Run base DP without any guidance (for comparison)."""
        obs_cond = self._encode_obs(image_data, qpos_data)
        B = obs_cond.shape[0]

        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device
        )
        naction = noisy_action

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for k in self.noise_scheduler.timesteps:
            noise_pred = self.nets["noise_pred_net"](
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        return naction.detach()


def load_guided_dp(
    dp_ckpt: str,
    tfm_ckpt: str,
    tfs_ckpt: str,
    alignment_ckpt: str,
    camera_names: list,
    pred_horizon: int = 20,
    action_dim: int = 7,
    horizons: list = None,
    guidance_scale: float = 0.1,
    guidance_start_ratio: float = 0.5,
    tfm_horizon: int = 8,
    device: str = "cuda",
) -> GuidedDiffusionPolicy:
    """Convenience function to load all components and build guided DP.

    Args:
        dp_ckpt: path to saved base DP nets (ModuleDict state)
        tfm_ckpt: path to TFM prediction head checkpoint
        tfs_ckpt: path to TFS checkpoint
        alignment_ckpt: path to VT alignment checkpoint
        camera_names: list of camera names
        pred_horizon: action chunk length
        action_dim: action dimension
        horizons: TFM prediction horizons
        guidance_scale: gradient guidance strength
        guidance_start_ratio: fraction of denoising with guidance (from end)
        tfm_horizon: which horizon to use for TFM prediction
        device: compute device
    """
    if horizons is None:
        horizons = [4, 8, 12]

    # Load base DP
    dp_state = torch.load(dp_ckpt, map_location="cpu")
    nets = nn.ModuleDict(dp_state).to(device)

    # Load TFM
    tfm = TactileForesightModel(
        alignment_ckpt=alignment_ckpt,
        horizons=horizons,
        device=device,
    )
    tfm.pred_head.load_state_dict(
        torch.load(tfm_ckpt, map_location="cpu")
    )

    # Load TFS
    tfs = TactileFeasibilityScore(
        alignment_ckpt=alignment_ckpt,
        action_dim=action_dim,
        chunk_size=pred_horizon,
        device=device,
    )
    tfs.load(tfs_ckpt)

    return GuidedDiffusionPolicy(
        nets=nets,
        tfm=tfm,
        tfs=tfs,
        camera_names=camera_names,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        guidance_scale=guidance_scale,
        guidance_start_ratio=guidance_start_ratio,
        horizon=tfm_horizon,
        device=device,
    )
