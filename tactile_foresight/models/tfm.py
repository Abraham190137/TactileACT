"""
Tactile Foresight Model (TFM) v2 — Sequence Prediction

Predicts a SEQUENCE of future tactile embeddings (t+1 to t+H) from current
visual (+ optional current tactile) observations in DINOv2 768-dim space.

Pipeline:
    1. Precomputed DINOv2 features (768-dim) as input
    2. Trainable Transformer prediction head → H predicted future tactile (768-dim each)

Training objective: MSE loss averaged over H steps:
    L = (1/H) Σ_{i=1}^{H} || T̂_{t+i} - DINOv2(T_{t+i}) ||^2
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFMPredictionHead(nn.Module):
    """Transformer-based prediction head for future tactile sequence.

    Predicts H steps of future tactile embeddings in DINOv2 768-dim space.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 4,
        nheads: int = 8,
        pred_horizon: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon

        # Project DINOv2 features to hidden_dim
        self.vision_proj = nn.Linear(embed_dim, hidden_dim)
        self.tactile_proj = nn.Linear(embed_dim, hidden_dim)

        # Learnable mask token (used when current tactile is masked)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learnable query tokens for each future timestep
        self.query_tokens = nn.Parameter(torch.randn(1, pred_horizon, hidden_dim) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Output projection back to DINOv2 space (shared across timesteps)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

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
        tactile_feat: Optional[torch.Tensor],
        tac_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vision_feat: (B, 768) DINOv2 vision feature
            tactile_feat: (B, 768) DINOv2 tactile feature, or None
            tac_mask: (B,) bool tensor, True = mask current tactile

        Returns:
            (B, H, 768) predicted future tactile sequence in DINOv2 space
        """
        B = vision_feat.shape[0]

        v = self.vision_proj(vision_feat).unsqueeze(1)  # (B, 1, hidden)

        if tactile_feat is not None and tac_mask is not None:
            t_proj = self.tactile_proj(tactile_feat).unsqueeze(1)
            mask_expanded = tac_mask.unsqueeze(1).unsqueeze(2)
            t = torch.where(mask_expanded, self.mask_token.expand(B, -1, -1), t_proj)
        elif tactile_feat is not None:
            t = self.tactile_proj(tactile_feat).unsqueeze(1)
        else:
            t = self.mask_token.expand(B, -1, -1)

        memory = torch.cat([v, t], dim=1)  # (B, 2, hidden)
        query = self.query_tokens.expand(B, -1, -1)  # (B, H, hidden)

        out = self.decoder(query, memory)  # (B, H, hidden)
        out = self.decoder_norm(out)
        pred = self.output_proj(out)  # (B, H, 768)
        return pred


class TactileForesightModel(nn.Module):
    """TFM v2: predicts future tactile SEQUENCE in DINOv2 768-dim space.

    Usage:
        tfm = TactileForesightModel(pred_horizon=20, device="cuda")

        # Training
        loss = tfm.compute_loss(v_feat, t_cur_feat, t_fut_seq, tac_mask)

        # Inference
        pred_seq = tfm.predict(v_feat, t_cur_feat, tac_mask)  # (B, 20, 768)
    """

    def __init__(
        self,
        dino_dim: int = 768,
        pred_horizon: int = 20,
        hidden_dim: int = 512,
        num_layers: int = 4,
        nheads: int = 8,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.dino_dim = dino_dim

        self.pred_head = TFMPredictionHead(
            embed_dim=dino_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nheads=nheads,
            pred_horizon=pred_horizon,
            dropout=dropout,
        )

        self.to(device)

    def predict(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: Optional[torch.Tensor],
        tac_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future tactile sequence in DINOv2 space.

        Args:
            vision_feat: (B, 768) precomputed DINOv2 vision feature
            tactile_feat: (B, 768) precomputed DINOv2 tactile feature, or None
            tac_mask: (B,) bool, True = mask current tactile

        Returns: (B, H, 768) predicted future tactile sequence
        """
        return self.pred_head(vision_feat, tactile_feat, tac_mask)

    def compute_loss(
        self,
        vision_feat: torch.Tensor,
        tactile_current_feat: torch.Tensor,
        tactile_future_seq: torch.Tensor,
        tac_mask: torch.Tensor,
    ) -> dict:
        """MSE loss averaged over H future steps.

        Args:
            vision_feat: (B, 768)
            tactile_current_feat: (B, 768)
            tactile_future_seq: (B, H, 768) target future tactile sequence
            tac_mask: (B,) bool

        Returns: dict with 'loss', 'mse_loss', 'cosine_sim'
        """
        t_pred = self.predict(vision_feat, tactile_current_feat, tac_mask)
        # t_pred: (B, H, 768), tactile_future_seq: (B, H, 768)

        mse_loss = F.mse_loss(t_pred, tactile_future_seq)

        with torch.no_grad():
            # Cosine sim averaged over all steps
            cos_sim = F.cosine_similarity(
                t_pred.reshape(-1, self.dino_dim),
                tactile_future_seq.reshape(-1, self.dino_dim),
                dim=-1,
            ).mean()

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
            "cosine_sim": cos_sim,
        }

    def trainable_parameters(self):
        return self.pred_head.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.pred_head.parameters() if p.requires_grad)

    def save_model(self, path: str):
        torch.save({
            "pred_head": self.pred_head.state_dict(),
            "dino_dim": self.dino_dim,
            "pred_horizon": self.pred_horizon,
        }, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.pred_head.load_state_dict(ckpt["pred_head"])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building TFM v2 on {device}...")

    tfm = TactileForesightModel(pred_horizon=20, device=device)
    print(f"Trainable parameters: {tfm.num_trainable_params():,}")

    B = 4
    v_feat = torch.randn(B, 768, device=device)
    t_cur = torch.randn(B, 768, device=device)
    t_fut_seq = torch.randn(B, 20, 768, device=device)
    mask = torch.tensor([True, False, True, False], device=device)

    losses = tfm.compute_loss(v_feat, t_cur, t_fut_seq, mask)
    print(f"MSE loss: {losses['mse_loss'].item():.4f}")
    print(f"Cosine sim: {losses['cosine_sim'].item():.4f}")

    pred = tfm.predict(v_feat, t_cur, mask)
    print(f"Predicted shape: {pred.shape}")  # (4, 20, 768)
    print("TFM v2 OK!")
