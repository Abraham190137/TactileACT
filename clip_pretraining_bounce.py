import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm


def replace_submodules(
    root_module: nn.Module,
    predicate,
    func,
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // features_per_group),
            num_channels=x.num_features,
        ),
    )
    return root_module


def modified_resnet18(features_per_group: int = 16) -> nn.Module:
    resnet18 = getattr(torchvision.models, "resnet18")()
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])  # remove avgpool+fc
    resnet18 = replace_bn_with_gn(resnet18, features_per_group=features_per_group)
    return resnet18


class ClipProjectionHead(nn.Module):
    def __init__(
        self,
        out_dim: int,
        conditioning_dim: int = 0,
        num_channels: int = 512,
        normalize: bool = True,
    ):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)
        self.linear = nn.Linear(num_channels + conditioning_dim, out_dim)
        self.normalize = normalize

    def forward(self, feature_map: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pooling(feature_map)
        x = self.flatten(x)
        if conditioning is not None:
            x = torch.cat((x, conditioning), dim=-1)
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


def clip_loss(
    image_embeddings: torch.Tensor,
    tac_embeddings: torch.Tensor,
    target_matrix: torch.Tensor,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    """
    image_embeddings: (B, N, Cams, D)
    tac_embeddings:   (B, N, D)
    target_matrix:    (N, N) identity
    returns:          (Cams,) per-camera loss
    """
    n_cameras = image_embeddings.shape[2]
    batch_size = image_embeddings.shape[0]

    image_embeddings = image_embeddings.permute(0, 2, 1, 3)  # (B, Cams, N, D)
    tac_embeddings = tac_embeddings.unsqueeze(1)  # (B, 1, N, D)
    image_logits = logit_scale * image_embeddings @ tac_embeddings.permute(0, 1, 3, 2)  # (B,Cams,N,N)
    tac_logits = logit_scale * tac_embeddings @ image_embeddings.permute(0, 1, 3, 2)  # (B,Cams,N,N)

    image_logits = image_logits.flatten(0, 1)  # (B*Cams, N, N)
    tac_logits = tac_logits.flatten(0, 1)

    image_loss = F.cross_entropy(
        image_logits, target_matrix.repeat(image_logits.shape[0], 1, 1), reduction="none"
    ).mean(dim=1)
    tac_loss = F.cross_entropy(
        tac_logits, target_matrix.T.repeat(tac_logits.shape[0], 1, 1), reduction="none"
    ).mean(dim=1)

    image_loss = image_loss.view(batch_size, n_cameras)
    tac_loss = tac_loss.view(batch_size, n_cameras)
    return ((image_loss + tac_loss) / 2.0).mean(dim=0)


@dataclass(frozen=True)
class BounceSchema:
    camera_names: Tuple[str, ...] = ("global", "wrist")
    tac_side: str = "left"  # "left" or "right"
    tac_key: str = "img"    # "img" (uint8 RGB) or "depth" (float32 single channel)
    proprio_key: str = "proprio_eef"  # "proprio_eef" or "proprio_joint"


class BounceClipDataset(torch.utils.data.Dataset):
    """
    Bounce dataset schema:
      /observations/images/{global,wrist} : (T,H,W,3) uint8
      /observations/tac/{left,right}/img  : (T,240,240,3) uint8   (or depth: (T,240,240))
      /observations/proprio_eef           : (T,6) float32  (or proprio_joint: (T,7))
    """

    def __init__(
        self,
        episode_ids: List[int],
        dataset_dir: str,
        schema: BounceSchema,
        proprio_mean: np.ndarray,
        proprio_std: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        min_distance: int = 5,
        n_images: int = 5,
    ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.schema = schema
        self.min_distance = int(min_distance)
        self.n_images = int(n_images)
        self.image_size = image_size
        self.n_cameras = len(schema.camera_names)

        self.proprio_mean = proprio_mean.astype(np.float32)
        self.proprio_std = np.clip(proprio_std.astype(np.float32), 1e-6, np.inf)

        self.rgb_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.episode_lengths: List[int] = []
        for episode_id in self.episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
            with h5py.File(dataset_path, "r") as f:
                # prefer infer T from images
                cam0 = schema.camera_names[0]
                T = f["observations"]["images"][cam0].shape[0]
                self.episode_lengths.append(int(T))
                if self.image_size is None:
                    H = f["observations"]["images"][cam0].shape[1]
                    W = f["observations"]["images"][cam0].shape[2]
                    self.image_size = (int(H), int(W))

        for length in self.episode_lengths:
            assert length >= self.n_images * self.min_distance * 1.2, (
                f"episode too short: length={length}, n_images={self.n_images}, min_distance={self.min_distance}"
            )

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int):
        episode_id = self.episode_ids[index]
        T = self.episode_lengths[index]

        timesteps: List[int] = []
        while len(timesteps) < self.n_images:
            t = int(np.random.randint(0, T))
            if all(abs(t - prev) >= self.min_distance for prev in timesteps):
                timesteps.append(t)

        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as f:
            obs = f["observations"]

            all_cam_images: List[torch.Tensor] = []
            all_tac: List[torch.Tensor] = []
            all_pos: List[torch.Tensor] = []

            for t in timesteps:
                # cameras
                timestep_cam_images: List[torch.Tensor] = []
                for cam_name in self.schema.camera_names:
                    img = obs["images"][cam_name][t]  # (H,W,3) uint8
                    img = torch.tensor(img, dtype=torch.float32) / 255.0
                    img = torch.einsum("h w c -> c h w", img)
                    img = self.rgb_normalize(img)
                    timestep_cam_images.append(img)
                cams = torch.stack(timestep_cam_images, dim=0)  # (Cams,3,H,W)

                # tactile
                tac_obj = obs["tac"][self.schema.tac_side][self.schema.tac_key][t]
                if self.schema.tac_key == "img":
                    tac = torch.tensor(tac_obj, dtype=torch.float32) / 255.0  # (Ht,Wt,3)
                    tac = torch.einsum("h w c -> c h w", tac)
                    tac = self.rgb_normalize(tac)
                elif self.schema.tac_key == "depth":
                    tac = torch.tensor(tac_obj, dtype=torch.float32)  # (Ht,Wt)
                    tac = tac.unsqueeze(0)  # (1,H,W)
                else:
                    raise ValueError(f"Unsupported tac_key: {self.schema.tac_key}")

                # proprio (condition on first 3 dims)
                pos = obs[self.schema.proprio_key][t].astype(np.float32)
                pos = (pos - self.proprio_mean) / self.proprio_std
                pos = torch.tensor(pos[:3], dtype=torch.float32)

                all_cam_images.append(cams)
                all_tac.append(tac)
                all_pos.append(pos)

        # (N, Cams, 3, H, W), (N, Ctac, Ht, Wt), (N, 3)
        return torch.stack(all_cam_images, dim=0), torch.stack(all_tac, dim=0), torch.stack(all_pos, dim=0)


def compute_proprio_stats(dataset_dir: str, episode_ids: List[int], proprio_key: str) -> Tuple[np.ndarray, np.ndarray]:
    vals = []
    for eid in tqdm(episode_ids, desc="Compute proprio stats"):
        p = os.path.join(dataset_dir, f"episode_{eid}.hdf5")
        with h5py.File(p, "r") as f:
            x = f["observations"][proprio_key][()]  # (T,D)
        vals.append(x)
    x = np.concatenate(vals, axis=0).astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.clip(std, 1e-6, np.inf)
    return mean, std


def pretrain(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    n_epochs: int = 200,
    clip_dim: int = 512,
    features_per_group: int = 16,
    resnet_lr: float = 1e-5,
    projection_lr: float = 1e-4,
    save_freq: int = 25,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "graphs"), exist_ok=True)

    dataset: BounceClipDataset = train_loader.dataset  # type: ignore[assignment]
    n_cameras = dataset.n_cameras
    state_size = 3

    vision_encoder = modified_resnet18(features_per_group=features_per_group).to(device)
    vision_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    tac_encoder = modified_resnet18(features_per_group=features_per_group).to(device)
    tac_projection = ClipProjectionHead(out_dim=clip_dim, conditioning_dim=state_size).to(device)

    optim_params = [
        {"params": tac_encoder.parameters(), "lr": resnet_lr},
        {"params": tac_projection.parameters(), "lr": projection_lr},
        {"params": vision_encoder.parameters(), "lr": resnet_lr},
        {"params": vision_projection.parameters(), "lr": projection_lr},
    ]
    optimizer = torch.optim.Adam(optim_params)

    training_losses = np.empty((n_epochs, n_cameras), dtype=np.float32)
    val_losses = np.empty((n_epochs, n_cameras), dtype=np.float32)

    for epoch in tqdm(range(n_epochs), desc="Pretrain"):
        vision_encoder.train()
        vision_projection.train()
        tac_encoder.train()
        tac_projection.train()

        train_loss = np.zeros(n_cameras, dtype=np.float32)
        for images, tac, pos in train_loader:
            images = images.to(device)  # (B,N,Cams,3,H,W)
            tac = tac.to(device)        # (B,N,Ctac,Ht,Wt)
            pos = pos.to(device)        # (B,N,3)

            B, N, Cams = images.shape[0], images.shape[1], images.shape[2]

            imgs_flat = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
            img_emb = vision_projection(vision_encoder(imgs_flat)).view(B, N, Cams, clip_dim)

            tac_flat = tac.view(-1, tac.shape[2], tac.shape[3], tac.shape[4])
            pos_flat = pos.view(-1, pos.shape[2])
            tac_emb = tac_projection(tac_encoder(tac_flat), pos_flat).view(B, N, clip_dim)

            target = torch.eye(N, device=device)
            loss_per_cam = clip_loss(img_emb, tac_emb, target)

            optimizer.zero_grad()
            loss_per_cam.mean().backward()
            optimizer.step()

            train_loss += loss_per_cam.detach().cpu().numpy()

        training_losses[epoch] = train_loss / max(1, len(train_loader))

        vision_encoder.eval()
        vision_projection.eval()
        tac_encoder.eval()
        tac_projection.eval()

        vloss = np.zeros(n_cameras, dtype=np.float32)
        with torch.no_grad():
            for images, tac, pos in val_loader:
                images = images.to(device)
                tac = tac.to(device)
                pos = pos.to(device)

                B, N, Cams = images.shape[0], images.shape[1], images.shape[2]
                imgs_flat = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
                img_emb = vision_projection(vision_encoder(imgs_flat)).view(B, N, Cams, clip_dim)

                tac_flat = tac.view(-1, tac.shape[2], tac.shape[3], tac.shape[4])
                pos_flat = pos.view(-1, pos.shape[2])
                tac_emb = tac_projection(tac_encoder(tac_flat), pos_flat).view(B, N, clip_dim)

                target = torch.eye(N, device=device)
                loss_per_cam = clip_loss(img_emb, tac_emb, target)
                vloss += loss_per_cam.detach().cpu().numpy()

        val_losses[epoch] = vloss / max(1, len(val_loader))

        np.save(os.path.join(save_dir, "graphs", "training_losses.npy"), training_losses[: epoch + 1])
        np.save(os.path.join(save_dir, "graphs", "val_losses.npy"), val_losses[: epoch + 1])

        if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
            torch.save(vision_encoder.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_vision_encoder.pth"))
            torch.save(vision_projection.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_vision_projection.pth"))
            torch.save(tac_encoder.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_tac_encoder.pth"))
            torch.save(tac_projection.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_tac_projection.pth"))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=89)
    parser.add_argument("--start_episode", type=int, default=90)
    parser.add_argument("--camera_names", type=str, default="global,wrist")
    parser.add_argument("--tac_side", type=str, choices=("left", "right"), default="left")
    parser.add_argument("--tac_key", type=str, choices=("img", "depth"), default="img")
    parser.add_argument("--proprio_key", type=str, choices=("proprio_eef", "proprio_joint"), default="proprio_eef")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--n_clip_images", type=int, default=5)
    parser.add_argument("--min_distance", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--clip_dim", type=int, default=512)
    parser.add_argument("--resnet_lr", type=float, default=1e-5)
    parser.add_argument("--projection_lr", type=float, default=1e-4)
    parser.add_argument("--save_freq", type=int, default=25)
    args = parser.parse_args()

    camera_names = tuple([s.strip() for s in args.camera_names.split(",") if s.strip()])
    schema = BounceSchema(
        camera_names=camera_names,
        tac_side=args.tac_side,
        tac_key=args.tac_key,
        proprio_key=args.proprio_key,
    )

    episode_ids = list(range(args.start_episode, args.start_episode + args.num_episodes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proprio_mean, proprio_std = compute_proprio_stats(args.dataset_dir, episode_ids, args.proprio_key)

    # split
    train_ratio = 0.8
    shuffled = np.random.permutation(len(episode_ids))
    split = int(train_ratio * len(episode_ids))
    train_ids = [episode_ids[i] for i in shuffled[:split]]
    val_ids = [episode_ids[i] for i in shuffled[split:]]

    train_ds = BounceClipDataset(
        train_ids,
        args.dataset_dir,
        schema,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        n_images=args.n_clip_images,
        min_distance=args.min_distance,
    )
    val_ds = BounceClipDataset(
        val_ids,
        args.dataset_dir,
        schema,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        n_images=args.n_clip_images,
        min_distance=args.min_distance,
    )

    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )
    train_loader = DataLoader(train_ds, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, **common_loader_kwargs)

    # persist run metadata
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "run_stats.txt"), "w") as f:
        f.write(f"dataset_dir: {args.dataset_dir}\n")
        f.write(f"episode_ids: {episode_ids}\n")
        f.write(f"camera_names: {schema.camera_names}\n")
        f.write(f"tac_side: {schema.tac_side}\n")
        f.write(f"tac_key: {schema.tac_key}\n")
        f.write(f"proprio_key: {schema.proprio_key}\n")
        f.write(f"proprio_mean: {proprio_mean.tolist()}\n")
        f.write(f"proprio_std: {proprio_std.tolist()}\n")
        f.write(f"n_clip_images: {args.n_clip_images}\n")
        f.write(f"min_distance: {args.min_distance}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"n_epochs: {args.n_epochs}\n")

    pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        n_epochs=args.n_epochs,
        clip_dim=args.clip_dim,
        resnet_lr=args.resnet_lr,
        projection_lr=args.projection_lr,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()

