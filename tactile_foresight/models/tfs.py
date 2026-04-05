"""
Tactile Feasibility Score (TFS) — TouchGuide-style Contact Physical Model.

Contrastive learning: observation embedding vs action embedding.
Observation = [vision_current, tactile_current, tactile_future(20 steps)]
Action = action_chunk (20, action_dim)

Score = cosine_similarity(O_emb, A_emb), trained with bidirectional InfoNCE.

Training uses ground-truth future tactile; inference uses TFM-predicted future tactile.
Actions are corrupted with DDPM-schedule noise during training (noise pretraining).

Reference: TouchGuide (Zhang et al., 2026) Section IV-B, Equations 5-10.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    """Encode [vision_current, tactile_current, tactile_future_seq] → O_emb.

    Input tokens: [v_token, tc_token, tf_1, ..., tf_H]  (2+H tokens)
    → TransformerEncoder → mean pool → L2 normalize → O_emb
    """

    def __init__(
        self,
        dino_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 4,
        nheads: int = 8,
        pred_horizon: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon

        # Project DINOv2 features to hidden_dim
        self.vision_proj = nn.Linear(dino_dim, hidden_dim)
        self.tactile_current_proj = nn.Linear(dino_dim, hidden_dim)
        self.tactile_future_proj = nn.Linear(dino_dim, hidden_dim)

        # Learnable type embeddings (vision / current_tactile / future_tactile)
        self.type_embed = nn.Parameter(torch.randn(3, 1, hidden_dim) * 0.02)

        # Positional embedding for future tactile sequence
        self.future_pos_embed = nn.Parameter(
            torch.randn(1, pred_horizon, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        vision_feat: torch.Tensor,
        tactile_current: torch.Tensor,
        tactile_future: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_feat: (B, 768) current vision DINOv2 feature
            tactile_current: (B, 768) current tactile DINOv2 feature
            tactile_future: (B, H, 768) future tactile sequence (GT or TFM-predicted)

        Returns:
            (B, hidden_dim) L2-normalized observation embedding
        """
        B = vision_feat.shape[0]

        v = self.vision_proj(vision_feat).unsqueeze(1)                # (B, 1, D)
        tc = self.tactile_current_proj(tactile_current).unsqueeze(1)  # (B, 1, D)
        tf = self.tactile_future_proj(tactile_future)                 # (B, H, D)

        # Add type embeddings
        v = v + self.type_embed[0]
        tc = tc + self.type_embed[1]
        tf = tf + self.type_embed[2] + self.future_pos_embed

        tokens = torch.cat([v, tc, tf], dim=1)  # (B, 2+H, D)

        out = self.encoder(tokens)  # (B, 2+H, D)
        out = self.norm(out)

        # Mean pooling → L2 normalize
        pooled = out.mean(dim=1)  # (B, D)
        return F.normalize(pooled, dim=-1)


class ActionEncoder(nn.Module):
    """Encode action chunk → A_emb using 1D CNN (TouchGuide style).

    (B, H, action_dim) → transpose → Conv1D stack → pool → L2 normalize
    """

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: (B, H, action_dim)

        Returns:
            (B, hidden_dim) L2-normalized action embedding
        """
        x = action.transpose(1, 2)    # (B, action_dim, H)
        x = self.conv_stack(x)        # (B, hidden_dim, H)
        x = x.mean(dim=-1)            # (B, hidden_dim)
        x = self.output_proj(x)       # (B, hidden_dim)
        return F.normalize(x, dim=-1)


class TactileFeasibilityScore(nn.Module):
    """TFS: contrastive model scoring (observation, action) pairs.

    Operates on precomputed DINOv2 768-dim features (no image input).
    Training: bidirectional InfoNCE with noisy action augmentation.
    Inference: score = dot(O_emb, A_emb) as gradient signal for guided DP.

    Usage:
        tfs = TactileFeasibilityScore(device="cuda")

        # Training (with GT future tactile + noisy actions)
        loss = tfs.compute_loss(v_feat, tc_feat, tf_seq, action_noisy)

        # Inference (with TFM-predicted future tactile)
        score = tfs.score(v_feat, tc_feat, tf_pred, action)
    """

    def __init__(
        self,
        dino_dim: int = 768,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        nheads: int = 8,
        pred_horizon: int = 20,
        dropout: float = 0.1,
        init_temperature: float = 0.07,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self.obs_encoder = ObservationEncoder(
            dino_dim=dino_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nheads=nheads,
            pred_horizon=pred_horizon,
            dropout=dropout,
        )
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # Learnable temperature (log scale for stability)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temperature))
        )

        self.to(device)

    @property
    def temperature(self) -> torch.Tensor:
        """Returns 1/τ (the logit scale), clamped for stability."""
        return self.log_temperature.exp().clamp(max=20.0)

    def encode_observation(
        self,
        vision_feat: torch.Tensor,
        tactile_current: torch.Tensor,
        tactile_future: torch.Tensor,
    ) -> torch.Tensor:
        """Returns L2-normalized observation embedding (B, D)."""
        return self.obs_encoder(vision_feat, tactile_current, tactile_future)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized action embedding (B, D)."""
        return self.action_encoder(action)

    def score(
        self,
        vision_feat: torch.Tensor,
        tactile_current: torch.Tensor,
        tactile_future: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feasibility score for each (obs, action) pair.

        Returns: (B,) scores (higher = more feasible)
        """
        o_emb = self.encode_observation(vision_feat, tactile_current, tactile_future)
        a_emb = self.encode_action(action)
        return (o_emb * a_emb).sum(dim=-1)  # cosine sim (both L2-normed)

    def compute_loss(
        self,
        vision_feat: torch.Tensor,
        tactile_current: torch.Tensor,
        tactile_future: torch.Tensor,
        action: torch.Tensor,
    ) -> dict:
        """Bidirectional InfoNCE loss (TouchGuide Eq. 8-10).

        Positive pairs: (O_i, A_i) from same timestep.
        Negatives: in-batch negatives (other samples in the batch).

        Args:
            vision_feat: (B, 768)
            tactile_current: (B, 768)
            tactile_future: (B, H, 768) ground-truth future tactile
            action: (B, H, action_dim) action chunk (may be noisy)

        Returns:
            dict with 'loss', 'loss_o2a', 'loss_a2o', 'accuracy', 'temperature'
        """
        o_emb = self.encode_observation(vision_feat, tactile_current, tactile_future)
        a_emb = self.encode_action(action)

        # Similarity matrix scaled by learned temperature
        logit_scale = self.temperature
        logits = logit_scale * (o_emb @ a_emb.t())  # (B, B)

        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        # Bidirectional InfoNCE
        loss_o2a = F.cross_entropy(logits, labels)
        loss_a2o = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_o2a + loss_a2o)

        # Monitoring metrics
        with torch.no_grad():
            pred_o2a = logits.argmax(dim=1)
            pred_a2o = logits.argmax(dim=0)
            acc = 0.5 * (
                (pred_o2a == labels).float().mean()
                + (pred_a2o == labels).float().mean()
            )

        return {
            "loss": loss,
            "loss_o2a": loss_o2a,
            "loss_a2o": loss_a2o,
            "accuracy": acc,
            "temperature": logit_scale,
        }

    def trainable_parameters(self):
        return self.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, path: str):
        torch.save({
            "obs_encoder": self.obs_encoder.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "log_temperature": self.log_temperature.data,
            "hidden_dim": self.hidden_dim,
            "pred_horizon": self.pred_horizon,
            "action_dim": self.action_dim,
        }, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.log_temperature.data = ckpt["log_temperature"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Building TFS on {}...".format(device))

    tfs = TactileFeasibilityScore(pred_horizon=20, device=device)
    print("Trainable parameters: {:,}".format(tfs.num_trainable_params()))

    B = 16
    v = torch.randn(B, 768, device=device)
    tc = torch.randn(B, 768, device=device)
    tf = torch.randn(B, 20, 768, device=device)
    a = torch.randn(B, 20, 7, device=device)

    result = tfs.compute_loss(v, tc, tf, a)
    print("Loss: {:.4f}".format(result["loss"].item()))
    print("Accuracy: {:.4f}".format(result["accuracy"].item()))
    print("Temperature (logit_scale): {:.4f}".format(result["temperature"].item()))

    scores = tfs.score(v, tc, tf, a)
    print("Scores shape: {}, mean: {:.4f}".format(scores.shape, scores.mean().item()))
    print("TFS OK!")
