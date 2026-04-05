import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    visualize: bool = False,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    image_embeddings: (B, N, Cams, D)
    tac_embeddings:   (B, N, D)
    target_matrix:    (N, N) identity
    returns:          (Cams,) per-camera loss, and optionally (Cams, N, N) similarity maps
    """
    n_cameras = image_embeddings.shape[2]
    batch_size = image_embeddings.shape[0]

    image_embeddings = image_embeddings.permute(0, 2, 1, 3)  # (B, Cams, N, D)
    tac_embeddings = tac_embeddings.unsqueeze(1)  # (B, 1, N, D)
    image_logits = logit_scale * image_embeddings @ tac_embeddings.permute(0, 1, 3, 2)  # (B,Cams,N,N)
    tac_logits = logit_scale * tac_embeddings @ image_embeddings.permute(0, 1, 3, 2)  # (B,Cams,N,N)

    sim_maps = None
    if visualize:
        # take first sample in batch, raw cosine similarity (before logit_scale)
        sim_maps = image_logits[0].clone().detach().cpu().numpy() / logit_scale  # (Cams, N, N)

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
    return ((image_loss + tac_loss) / 2.0).mean(dim=0), sim_maps


@dataclass(frozen=True)
class BounceSchema:
    camera_names: Tuple[str, ...] = ("global", "wrist")
    tac_side: str = "left"  # "left" or "right"
    tac_key: str = "img"    # "img" (uint8 RGB) or "depth" (float32 single channel)
    proprio_key: str = "proprio_eef"  # "proprio_eef" or "proprio_joint"


def _sample_spaced_timesteps(T: int, n: int, min_distance: int) -> List[int]:
    """
    Deterministic segment-based sampling: divide [0, T) into n segments,
    then sample one random timestep from each segment with min_distance guarantee.
    Falls back to rejection sampling only if segment approach fails (shouldn't happen
    when T >= n * min_distance * 1.2).
    """
    # each segment needs at least min_distance width to guarantee spacing
    segment_len = T / n
    timesteps: List[int] = []
    for i in range(n):
        lo = int(i * segment_len)
        hi = int((i + 1) * segment_len)
        # clamp to valid range
        lo = max(lo, 0)
        hi = min(hi, T)
        t = int(np.random.randint(lo, hi))
        timesteps.append(t)
    # verify min_distance; sort for check
    timesteps.sort()
    valid = all(timesteps[j] - timesteps[j - 1] >= min_distance for j in range(1, len(timesteps)))
    if valid:
        return timesteps
    # fallback: greedy sorted sampling with guaranteed spacing
    timesteps = []
    available = T - (n - 1) * min_distance  # slack
    if available <= 0:
        # extreme edge case: just space evenly
        return [int(i * min_distance) for i in range(n)]
    offsets = sorted(np.random.choice(available, size=n, replace=False))
    for i, off in enumerate(offsets):
        timesteps.append(int(off + i * min_distance))
    return timesteps


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
            with h5py.File(dataset_path, "r", swmr=True) as f:
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

        timesteps = _sample_spaced_timesteps(T, self.n_images, self.min_distance)

        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r", swmr=True) as f:
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
        with h5py.File(p, "r", swmr=True) as f:
            x = f["observations"][proprio_key][()]  # (T,D)
        vals.append(x)
    x = np.concatenate(vals, axis=0).astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.clip(std, 1e-6, np.inf)
    return mean, std


def _plot_similarity_maps(sim_maps: np.ndarray, epoch: int, save_dir: str, split: str, camera_names: Tuple[str, ...]):
    """
    sim_maps: (Cams, N, N) cosine similarity matrix
    Saves one heatmap per camera.
    """
    for cam_idx in range(sim_maps.shape[0]):
        cam_name = camera_names[cam_idx] if cam_idx < len(camera_names) else str(cam_idx)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(sim_maps[cam_idx], cmap="viridis", aspect="equal")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Similarity Map - Epoch {epoch}, Cam {cam_name} ({split})")
        ax.set_xlabel("Tactile index")
        ax.set_ylabel("Image index")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "graphs", f"epoch_{epoch}_cam_{cam_name}_{split}.png"), dpi=100)
        plt.close(fig)


def _plot_loss_curves(training_losses: np.ndarray, val_losses: np.ndarray, epoch: int, n_epochs: int, save_dir: str, camera_names: Tuple[str, ...]):
    """
    training_losses, val_losses: (epoch+1, n_cameras)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(training_losses.shape[1]):
        cam_name = camera_names[i] if i < len(camera_names) else str(i)
        ax.plot(training_losses[:, i], label=f"{cam_name} train", color=f"C{i}")
        ax.plot(val_losses[:, i], label=f"{cam_name} val", linestyle="dashed", color=f"C{i}")
    ax.legend(loc="best")
    ax.set_title(f"Training and Validation Loss - Epoch {epoch + 1}/{n_epochs}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "graphs", "training_loss.png"), dpi=100)
    plt.close(fig)


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
    plot_freq: int = 50,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "graphs"), exist_ok=True)

    dataset: BounceClipDataset = train_loader.dataset  # type: ignore[assignment]
    n_cameras = dataset.n_cameras
    camera_names = dataset.schema.camera_names
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

    training_losses = np.empty((n_epochs, n_cameras), dtype=np.float32)
    val_losses = np.empty((n_epochs, n_cameras), dtype=np.float32)

    for epoch in tqdm(range(n_epochs), desc="Pretrain"):
        vision_encoder.train()
        vision_projection.train()
        tac_encoder.train()
        tac_projection.train()

        train_loss = np.zeros(n_cameras, dtype=np.float32)
        should_visualize = (epoch % plot_freq == 0)

        for batch_idx, (images, tac, pos) in enumerate(train_loader):
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
            do_vis = (batch_idx == 0 and should_visualize)
            loss_per_cam, sim_maps = clip_loss(img_emb, tac_emb, target, visualize=do_vis)

            if do_vis and sim_maps is not None:
                _plot_similarity_maps(sim_maps, epoch, save_dir, "train", camera_names)

            optimizer.zero_grad()
            loss_per_cam.mean().backward()
            optimizer.step()

            train_loss += loss_per_cam.detach().cpu().numpy()

        training_losses[epoch] = train_loss / max(1, len(train_loader))

        # --- validation ---
        vision_encoder.eval()
        vision_projection.eval()
        tac_encoder.eval()
        tac_projection.eval()

        vloss = np.zeros(n_cameras, dtype=np.float32)
        with torch.no_grad():
            for batch_idx, (images, tac, pos) in enumerate(val_loader):
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
                do_vis = (batch_idx == 0 and should_visualize)
                loss_per_cam, sim_maps = clip_loss(img_emb, tac_emb, target, visualize=do_vis)

                if do_vis and sim_maps is not None:
                    _plot_similarity_maps(sim_maps, epoch, save_dir, "val", camera_names)

                vloss += loss_per_cam.detach().cpu().numpy()

        val_losses[epoch] = vloss / max(1, len(val_loader))

        # step LR scheduler
        scheduler.step()

        # save loss npy files every epoch
        np.save(os.path.join(save_dir, "graphs", "training_losses.npy"), training_losses[: epoch + 1])
        np.save(os.path.join(save_dir, "graphs", "val_losses.npy"), val_losses[: epoch + 1])

        # plot loss curves periodically
        if should_visualize:
            _plot_loss_curves(training_losses[: epoch + 1], val_losses[: epoch + 1], epoch, n_epochs, save_dir, camera_names)

        # save model checkpoints
        if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
            torch.save(vision_encoder.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_vision_encoder.pth"))
            torch.save(vision_projection.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_vision_projection.pth"))
            torch.save(tac_encoder.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_tac_encoder.pth"))
            torch.save(tac_projection.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_tac_projection.pth"))

        # log current LR to tqdm
        current_lr = scheduler.get_last_lr()
        tqdm.write(f"[Epoch {epoch}] train_loss={training_losses[epoch].mean():.4f}  val_loss={val_losses[epoch].mean():.4f}  lr={current_lr[0]:.2e}")


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
    parser.add_argument("--plot_freq", type=int, default=50)
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

    # split (固定seed确保每次划分一致)
    train_ratio = 0.8
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(len(episode_ids))
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
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)

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
        f.write(f"resnet_lr: {args.resnet_lr}\n")
        f.write(f"projection_lr: {args.projection_lr}\n")
        f.write(f"plot_freq: {args.plot_freq}\n")
        f.write(f"train_ids: {sorted(train_ids)}\n")
        f.write(f"val_ids: {sorted(val_ids)}\n")

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
        plot_freq=args.plot_freq,
    )


if __name__ == "__main__":
    main()
