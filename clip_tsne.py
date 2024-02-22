import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from clip_pretraining import modified_resnet18, ClipProjectionHead, ClipDataset
from utils import get_norm_stats
import torch
import h5py
from tqdm import tqdm
from typing import List

def plot_tsne(all_image_vectors:np.ndarray, gelsight_vectors:np.ndarray, timestamps):
    # all_image_vectors: (n_cam, episode_len, 512)
    n_cam = all_image_vectors.shape[0]
    episode_len = len(timestamps)
    tsne = TSNE(n_components=2, random_state=10)

    print(all_image_vectors[0].shape)
    print(gelsight_vectors.shape)

    all_latent_vectors = np.concatenate([all_image_vectors[i] for i in range(n_cam)] + [gelsight_vectors], axis=0)
    print(all_latent_vectors.shape)

    embedded = tsne.fit_transform(all_latent_vectors)
    image_embedings = [embedded[i*episode_len:(i+1)*episode_len] for i in range(n_cam)]
    gelsight_embedded = embedded[n_cam*episode_len:]

    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '*', 'D', 'P', 'H']
    for i, image_embedded in enumerate(image_embedings):
        plt.scatter(image_embedded[:, 0], image_embedded[:, 1], c=timestamps/len(timestamps), cmap='viridis', marker=markers[i], s=50, alpha=0.25, label=f'Image Vectors {i+1}')
    plt.scatter(gelsight_embedded[:, 0], gelsight_embedded[:, 1], c=timestamps/len(timestamps), cmap='viridis', marker='x', s=50, alpha=0.25, label='Gelsight Vectors')

    plt.title('t-SNE Visualization of Latent Vectors')
    # put legend outside of plot
    plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Timestamp')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    ACT_model_path = "/home/aigeorge/research/TactileACT/data/camera_cage/pretrained_1999_melted/policy_last.ckpt"
    image_encoder_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_vision_encoder.pth"
    image_projection_head_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_vision_projection.pth"
    gelsight_encoder_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_gelsight_encoder.pth"
    gelsight_projection_head_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_gelsight_projection.pth"
    dataset_dir = "/home/aigeorge/research/TactileACT/data/camera_cage/data"
    num_episodes = 39
    camera_names = ["1", "2", "3", "4", "5", "6"]
    use_act = False

    pretrained = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_act:
        from load_ACT import load_ACT
        act = load_ACT(ACT_model_path)
        vision_encoder = act.model.backbones[0][0]
        gelsight_encoder = act.model.backbones[1][0]

    else:
        # create the vision encoder and projection head
        vision_encoder = modified_resnet18().to(device)
        if pretrained:
            vision_encoder.load_state_dict(torch.load(image_encoder_path))

        # create the gelsight encoder and projection head
        gelsight_encoder = modified_resnet18().to(device)
        if pretrained:
            gelsight_encoder.load_state_dict(torch.load(gelsight_encoder_path))

    # vision_projection_head = ClipProjectionHead(out_dim=512).to(device)
    # if pretrained:
    #     vision_projection_head.load_state_dict(torch.load(image_projection_head_path))

    # gelsight_projection_head = ClipProjectionHead(out_dim=512, conditioning_dim=3).to(device)
    # if pretrained:
    #     gelsight_projection_head.load_state_dict(torch.load(gelsight_projection_head_path))
    vision_projection_head = torch.nn.AdaptiveMaxPool2d((1, 1))
    gelsight_projection_head = torch.nn.AdaptiveMaxPool2d((1, 1))
    
    vision_encoder.eval()
    vision_projection_head.eval()
    gelsight_encoder.eval()
    gelsight_projection_head.eval()

    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=True)
    dataset = ClipDataset(list(range(num_episodes)), dataset_dir, camera_names, norm_stats)

    episode_idx = 0
    episode_len = dataset.episode_lengths[episode_idx]

    all_image_vectors = np.empty((len(camera_names), episode_len, 512))
    gelsight_vectors = np.empty((episode_len, 512))
    timestamps = []

    with torch.no_grad():
        for t in tqdm(range(episode_len)):
            images = [dataset.get_image(episode_idx, t, cam) for cam in camera_names]
            gelsight = dataset.get_gelsight(episode_idx, t)
            position = dataset.get_position(episode_idx, t)

            for i, image in enumerate(images):
                image_vector = vision_projection_head(vision_encoder(image.to(device).unsqueeze(0)))
                all_image_vectors[i, t, :] = image_vector.detach().cpu().numpy().squeeze()

            gelsight_vector = gelsight_projection_head(gelsight_encoder(gelsight.to(device).unsqueeze(0))) #, position.to(device).unsqueeze(0))
            gelsight_vectors[t, :] = gelsight_vector.detach().cpu().numpy().squeeze()
            timestamps.append(t) # can be replaced with actual timestamps

    timestamps = np.array(timestamps)

    plot_tsne(all_image_vectors, gelsight_vectors, timestamps)