"""
Contact Physical Model (CPM) — Paper-faithful implementation following TouchGuide.

Key difference from TFS: observation encoder only uses CURRENT (V_t, T_t),
NO future tactile sequence. This matches the paper's Eq. 5-7.

Observation = concat(T_emb, V_emb) -> TransformerEncoder -> L2 Norm -> O_t
Action = action_chunk (H, action_dim) -> 1D CNN + MLP -> L2 Norm -> a_t
Score = O_t^T * a_t (cosine similarity)

Training: bidirectional InfoNCE with geometric-distribution noise pretraining.

Reference: TouchGuide (Zhang et al., 2026) Section IV-B, Equations 5-10.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    """Encode [vision_current, tactile_current] -> O_emb.

    Paper Eq. 5: O_t = TranEnc(concat(T_emb, V_emb)) -> L2 Norm
    Input tokens: [t_token, v_token]  (2 tokens)
    -> TransformerEncoder -> mean pool -> L2 normalize -> O_emb
    """

    def __init__(
        self,
        dino_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 4,
        nheads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project DINOv2 features to hidden_dim
        self.tactile_proj = nn.Linear(dino_dim, hidden_dim)
        self.vision_proj = nn.Linear(dino_dim, hidden_dim)

        # Learnable type embeddings (tactile / vision)
        self.type_embed = nn.Parameter(torch.randn(2, 1, hidden_dim) * 0.02)

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
        tactile_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_feat: (B, 768) current vision DINOv2 feature
            tactile_feat: (B, 768) current tactile DINOv2 feature

        Returns:
            (B, hidden_dim) L2-normalized observation embedding
        """
        # Paper: concat(T_emb, V_emb) — tactile first, then vision
        t = self.tactile_proj(tactile_feat).unsqueeze(1)  # (B, 1, D)
        v = self.vision_proj(vision_feat).unsqueeze(1)    # (B, 1, D)

        # Add type embeddings
        t = t + self.type_embed[0]
        v = v + self.type_embed[1]

        tokens = torch.cat([t, v], dim=1)  # (B, 2, D)

        out = self.encoder(tokens)  # (B, 2, D)
        out = self.norm(out)

        # Mean pooling -> L2 normalize
        pooled = out.mean(dim=1)  # (B, D)
        return F.normalize(pooled, dim=-1)


class ActionEncoder(nn.Module):
    """Encode action chunk -> A_emb using 1D CNN (TouchGuide style).

    Paper Eq. 6: a_t = E_A(A_t) -> L2 Norm
    (B, H, action_dim) -> transpose -> Conv1D stack -> pool -> L2 normalize
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


class ContactPhysicalModel(nn.Module):
    """CPM: contrastive model scoring (observation, action) pairs.

    Paper-faithful: observation = (V_t, T_t) only, no future tactile.

    Training: bidirectional InfoNCE with noisy action augmentation.
    Inference: score = dot(O_emb, A_emb) as gradient signal for guided DP.

    Usage:
        cpm = ContactPhysicalModel(device="cuda")

        # Training (with GT actions, some noisy)
        loss = cpm.compute_loss(v_feat, t_feat, action_noisy)

        # Inference (score for guidance)
        score = cpm.score(v_feat, t_feat, action)
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
        self.dino_dim = dino_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nheads = nheads
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.dropout = dropout

        self.obs_encoder = ObservationEncoder(
            dino_dim=dino_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nheads=nheads,
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
        """Returns 1/tau (the logit scale), clamped for stability."""
        return self.log_temperature.exp().clamp(max=100.0)

    def encode_observation(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Returns L2-normalized observation embedding (B, D)."""
        return self.obs_encoder(vision_feat, tactile_feat)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized action embedding (B, D)."""
        return self.action_encoder(action)

    def score(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feasibility score for each (obs, action) pair.

        Paper Eq. 7: s = O_t^T * a_t

        Returns: (B,) scores (higher = more feasible)
        """
        o_emb = self.encode_observation(vision_feat, tactile_feat)
        a_emb = self.encode_action(action)
        return (o_emb * a_emb).sum(dim=-1)

    def compute_loss(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: torch.Tensor,
        action: torch.Tensor,
    ) -> dict:
        """Bidirectional InfoNCE loss (TouchGuide Eq. 8-10).

        Positive pairs: (O_i, A_i) from same timestep.
        Negatives: in-batch negatives.

        Args:
            vision_feat: (B, 768)
            tactile_feat: (B, 768)
            action: (B, H, action_dim) action chunk (may be noisy)

        Returns:
            dict with 'loss', 'loss_o2a', 'loss_a2o', 'accuracy', 'temperature'
        """
        o_emb = self.encode_observation(vision_feat, tactile_feat)
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
            "dino_dim": self.dino_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "nheads": self.nheads,
            "pred_horizon": self.pred_horizon,
            "action_dim": self.action_dim,
            "dropout": self.dropout,
        }, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.log_temperature.data = ckpt["log_temperature"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Building CPM on {}...".format(device))

    cpm = ContactPhysicalModel(pred_horizon=20, device=device)
    print("Trainable parameters: {:,}".format(cpm.num_trainable_params()))

    B = 16
    v = torch.randn(B, 768, device=device)
    t = torch.randn(B, 768, device=device)
    a = torch.randn(B, 20, 7, device=device)

    result = cpm.compute_loss(v, t, a)
    print("Loss: {:.4f}".format(result["loss"].item()))
    print("Accuracy: {:.4f}".format(result["accuracy"].item()))
    print("Temperature (logit_scale): {:.4f}".format(result["temperature"].item()))

    scores = cpm.score(v, t, a)
    print("Scores shape: {}, mean: {:.4f}".format(scores.shape, scores.mean().item()))
    print("CPM OK!")
