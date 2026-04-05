"""
DINOv2 ViT-B/14 frozen feature extractor.

Provides a unified interface for extracting visual and tactile embeddings
using a pretrained DINOv2 model. The model is always frozen (no gradients).

Usage:
    extractor = DINOv2Extractor(device="cuda")
    # raw uint8 images (B, H, W, 3) or (B, 3, H, W) float
    emb = extractor(images)  # (B, 768)
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn
import torchvision.transforms as T


def _patch_dinov2_for_py38():
    """Patch cached DINOv2 source files for Python <3.10 compatibility.

    The latest DINOv2 uses PEP 604 union syntax (`float | None`) which is
    only supported at runtime from Python 3.10+.  Adding
    `from __future__ import annotations` makes the annotations lazy strings,
    avoiding the TypeError.
    """
    hub_dir = torch.hub.get_dir()
    pattern = os.path.join(hub_dir, "facebookresearch_dinov2_main", "dinov2", "**", "*.py")
    for path in glob.glob(pattern, recursive=True):
        with open(path, "r") as f:
            first_line = f.readline()
        if "from __future__ import annotations" not in first_line:
            with open(path, "r") as f:
                content = f.read()
            if " | " in content:  # only patch files that actually use union syntax
                with open(path, "w") as f:
                    f.write("from __future__ import annotations\n" + content)


class DINOv2Extractor(nn.Module):
    """Frozen DINOv2 ViT-B/14 feature extractor.

    Outputs the [CLS] token embedding (768-dim) for each input image.
    All parameters are frozen; this module never requires gradients
    for the backbone itself.
    """

    EMBED_DIM = 768  # ViT-B/14

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.model_name = model_name

        # Load via torch.hub; patch for Python 3.8 compat first
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True,
            )
        except TypeError:
            _patch_dinov2_for_py38()
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True,
            )
        self.backbone.eval()

        # Freeze all parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # DINOv2 expected preprocessing: 224x224, ImageNet norm
        # NOTE: if images are already ImageNet-normalized (from dataset),
        # use preprocess_resize_only to avoid double normalization.
        self._resize = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
        ])
        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.to(device)

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, already_normalized: bool = True
    ) -> torch.Tensor:
        """Extract DINOv2 CLS token embeddings.

        Args:
            images: (B, 3, H, W) float tensor.
            already_normalized: if True, skip ImageNet normalization
                (images already have mean/std applied by dataset).

        Returns:
            (B, 768) float tensor — CLS token embedding.
        """
        x = images.float()
        if x.max() > 200.0:  # raw uint8
            x = x / 255.0

        x = self._resize(x)
        if not already_normalized:
            x = self._normalize(x)
        features = self.backbone(x)  # (B, 768) CLS token
        return features

    @torch.no_grad()
    def extract_patch_tokens(
        self, images: torch.Tensor, already_normalized: bool = True
    ) -> torch.Tensor:
        """Extract all patch token embeddings (without CLS).

        Args:
            images: (B, 3, H, W) float tensor.
            already_normalized: if True, skip ImageNet normalization.

        Returns:
            (B, N_patches, 768) float tensor where N_patches = (224/14)^2 = 256.
        """
        x = images.float()
        if x.max() > 200.0:
            x = x / 255.0

        x = self._resize(x)
        if not already_normalized:
            x = self._normalize(x)
        out = self.backbone.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]  # (B, 256, 768)
        return patch_tokens

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 ViT-B/14 on {device}...")
    t0 = time.time()
    extractor = DINOv2Extractor(device=device)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Test with random images
    dummy = torch.randn(4, 3, 480, 640, device=device)
    t0 = time.time()
    emb = extractor(dummy)
    print(f"CLS embedding shape: {emb.shape}")  # (4, 768)
    print(f"Inference time: {(time.time() - t0)*1000:.1f}ms")

    patches = extractor.extract_patch_tokens(dummy)
    print(f"Patch tokens shape: {patches.shape}")  # (4, 256, 768)
