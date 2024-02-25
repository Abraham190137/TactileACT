import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from clip_pretraining import modified_resnet18, ClipProjectionHead, ClipDataset
from utils import get_norm_stats
import torch
from torch.nn import functional as F
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


import cv2
def visualize_gelsight_data(image):
	# Convert the image to LAB color space
	max_depth = 10
	max_strain = 30
	# Show all three using LAB color space
	image[0] = np.clip(100*np.maximum(image[0], 0)/max_depth, 0, 100)
	# normalized_depth = np.clip(100*(depth_image/depth_image.max()), 0, 100)
	image[1:] = np.clip(128*(image[1:]/max_strain), -128, 127)
	return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

def show_scene(data_file, timestep) -> None:
    print('showing images in', data_file, 'at timestep', timestep)
    with h5py.File(data_file, 'r') as f:
        # tile the images (2x3) plus gelsight
        image_size = (f.attrs['image_height'], f.attrs['image_width'])
        gelsight_size = (f.attrs['gelsight_height'], f.attrs['gelsight_width'])
        grid = np.zeros([2*image_size[0] + gelsight_size[0], 3*image_size[1], 3], dtype=np.uint8)
        for i, key in enumerate(f['observations/images'].keys()):
            image_data = f['observations/images'][key][timestep, :, :, :]
            grid[(i//3)*image_size[0]:(i//3+1)*image_size[0], (i%3)*image_size[1]:(i%3+1)*image_size[1], :] = image_data
        gelsight_data = f['observations/gelsight/depth_strain_image'][timestep, :, :, :]
        gelsight_data = visualize_gelsight_data(gelsight_data)*255
        grid[image_size[0]*2:, :gelsight_size[1], :] = gelsight_data
        cv2.imshow('Images', grid)
        cv2.waitKey(0)

def test_confidence_score(all_image_vectors, all_gelsight_vectors, dataset_dir, episode_idx, episode_len):
    for t in range(episode_len):
        if t != 0:
            print("confidence: ", np.dot(all_gelsight_vectors[t-1], all_gelsight_vectors[t])/np.linalg.norm(all_gelsight_vectors[t-1])/np.linalg.norm(all_gelsight_vectors[t]))
        show_scene(f'{dataset_dir}/episode_{episode_idx}.hdf5', t)

def plot_run_similarity(all_vectors, name):
    episode_len = all_vectors.shape[0]
    similarity = np.zeros((episode_len, episode_len))
    for i in range(episode_len):
        for j in range(episode_len):
            similarity[i, j] = np.dot(all_vectors[i], all_vectors[j])/np.linalg.norm(all_vectors[i])/np.linalg.norm(all_vectors[j])

    plt.figure()
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.title('Similarity Matrix for ' + name)
    plt.xlabel('Timestep')
    plt.ylabel('Timestep')

    # differences_1 = np.zeros(episode_len)
    # for i in range(1, episode_len):
    #     differences_1[i] = 1 - (similarity[i-1, i])**4

    # differences_5 = np.zeros(episode_len)
    # for i in range(5, episode_len):
    #     differences_5[i] = 1 - similarity[i-5, i]

    # plt.figure()
    # plt.plot(differences_1, label='1 timestep difference')
    # # plt.plot(differences_5, label='5 timestep difference')
    # plt.title('Difference in Similarity for ' + name)
    



if __name__ == "__main__":
    image_encoder_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/11/epoch_1499_vision_encoder.pth"
    image_projection_head_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/11/epoch_1499_vision_projection.pth"
    gelsight_encoder_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/11/epoch_1499_gelsight_encoder.pth"
    gelsight_projection_head_path = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/11/epoch_1499_gelsight_projection.pth"
    dataset_dir = "/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/data"
    num_episodes = 101

    # ACT_model_path = "/home/aigeorge/research/TactileACT/data/camera_cage/pretrained_1999_melted/policy_last.ckpt"
    # image_encoder_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_vision_encoder.pth"
    # image_projection_head_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_vision_projection.pth"
    # gelsight_encoder_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_gelsight_encoder.pth"
    # gelsight_projection_head_path = "/home/aigeorge/research/TactileACT/clip_models/4/epoch_1999_gelsight_projection.pth"
    # dataset_dir = "/home/aigeorge/research/TactileACT/data/camera_cage/data"
    # num_episodes = 39

    camera_names = ["1", "2", "3", "4", "5", "6"]

    encoder_pretrained = True
    projection_head_pretrained = True
    use_projection_head = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the vision encoder and projection head
    vision_encoder = modified_resnet18().to(device)
    if encoder_pretrained:
        vision_encoder.load_state_dict(torch.load(image_encoder_path))

    # create the gelsight encoder and projection head
    gelsight_encoder = modified_resnet18().to(device)
    if encoder_pretrained:
        gelsight_encoder.load_state_dict(torch.load(gelsight_encoder_path))

    vision_projection_head = ClipProjectionHead(512).to(device)
    if projection_head_pretrained:
        vision_projection_head.load_state_dict(torch.load(image_projection_head_path))

    gelsight_projection_head = ClipProjectionHead(512, 3).to(device)
    if  projection_head_pretrained:
        gelsight_projection_head.load_state_dict(torch.load(gelsight_projection_head_path))

    # ACT_model_path = "/home/aigeorge/research/TactileACT/data/camera_cage/pretrained_1999_melted/policy_last.ckpt"
    # from load_ACT import load_ACT
    # act = load_ACT(ACT_model_path)
    # vision_encoder = act.model.backbones[0][0]
    # gelsight_encoder = act.model.backbones[1][0]

    vision_encoder.eval()
    vision_projection_head.eval()
    gelsight_encoder.eval()
    gelsight_projection_head.eval()

    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=True)
    dataset = ClipDataset(list(range(num_episodes)), dataset_dir, camera_names, norm_stats)

    # episode_idxs = [0, 3] # 0 is training, 3 is testing
    episode_idxs = [0]
    for episode_idx in episode_idxs:
        episode_len = dataset.episode_lengths[episode_idx]

        all_image_vectors = np.empty((len(camera_names), episode_len, 512))
        gelsight_vectors = np.empty((episode_len, 512))
        all_image_encodings = np.empty((len(camera_names), episode_len, 512))
        gelsight_encodings = np.empty((episode_len, 512))
        timestamps = []

        with torch.no_grad():
            for t in tqdm(range(episode_len)):
                images = [dataset.get_image(episode_idx, t, cam) for cam in camera_names]
                gelsight = dataset.get_gelsight(episode_idx, t)
                position = dataset.get_position(episode_idx, t)

                for i, image in enumerate(images):
                    encoding = vision_encoder(image.to(device).unsqueeze(0))
                    image_vector = vision_projection_head(encoding)
                    all_image_vectors[i, t, :] = image_vector.detach().cpu().numpy().squeeze()
                    encoding = F.adaptive_avg_pool2d(encoding, (1, 1))
                    all_image_encodings[i, t, :] = encoding.detach().cpu().numpy().squeeze()

                encoding = gelsight_encoder(gelsight.to(device).unsqueeze(0))
                gelsight_vector = gelsight_projection_head(encoding, position.to(device).unsqueeze(0))
                gelsight_vectors[t, :] = gelsight_vector.detach().cpu().numpy().squeeze()
                encoding = F.adaptive_avg_pool2d(encoding, (1, 1))
                gelsight_encodings[t, :] = encoding.detach().cpu().numpy().squeeze()
                timestamps.append(t) # can be replaced with actual timestamps

        timestamps = np.array(timestamps)
        if not use_projection_head:
            all_image_vectors = all_image_encodings
            gelsight_vectors = gelsight_encodings

        plot_run_similarity(gelsight_vectors, 'Gelsight')
        # plt.show()
        # exit()
        for i, embedding in enumerate(all_image_vectors):
            plot_run_similarity(embedding, 'camera ' + str(i+1))

        
        plot_tsne(all_image_vectors, gelsight_vectors, timestamps)
        plt.title(f't-SNE Visualization of Latent Vectors for Episode {episode_idx}')
        # plt.show()
        # test_confidence_score(all_image_vectors, gelsight_vectors, dataset_dir, episode_idx, episode_len)
        
    plt.show()