import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2
from torchvision import transforms
from tqdm import tqdm
import json

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size, image_size = None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        # since both actions and qpos are in the same coordinate frame and units, we use an average of the twos mean and std
        self.mean = (norm_stats["action_mean"] + norm_stats["qpos_mean"]) / 2
        # self.mean_matrix = mean.repeat(chunk_size, 1)
        # print("mean_matrix", self.mean_matrix.shape)
        self.std = (norm_stats["action_std"] + norm_stats["qpos_std"]) / 2
        # self.std_matrix = std.repeat(chunk_size, 1)
        # print("std_matrix", self.std_matrix.shape)
        self.is_sim = None
        self.image_size = image_size # image size in (H, W)
        if "gelsight_mean" in norm_stats: # if gelsight data exists
            self.gelsight_mean = norm_stats["gelsight_mean"]
            self.gelsight_std = norm_stats["gelsight_std"]

        self.chunk_size = chunk_size

        # image normalization for resnet. 
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.__getitem__(0) # initialize self.is_sim, self.image_size


        # image = normalize(image)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts] # unused

            if self.image_size is None: # if image size is not specified, use the saved image size
                self.image_size = (root.attrs['image_height'], root.attrs['image_width'])

            all_cam_images = []

            for cam_name in self.camera_names:
                # seperate processing for gelsight:
                if cam_name == 'gelsight':
                    gelsight_data = root['observations/gelsight/depth_strain_image'][start_ts]
                    # gelsight_data = cv2.resize(gelsight_data, (self.image_size[1], self.image_size[0]))
                    # adjust gelsight data using the mean and std
                    gelsight_data = (gelsight_data - self.gelsight_mean) / self.gelsight_std
                    gelsight_data = torch.tensor(gelsight_data, dtype=torch.float32)
                    gelsight_data = torch.einsum('h w c -> c h w', gelsight_data) # change to c h w
                    all_cam_images.append(gelsight_data)
                
                else:
                    image = root[f'/observations/images/{cam_name}'][start_ts]
                    # resize image
                    if self.image_size != image.shape[:2]:
                        print('reshaping image')
                        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

                    # normalize image
                    image = torch.tensor(image, dtype=torch.float32)/255.0
                    image = torch.einsum('h w c -> c h w', image) # change to c h w
                    image = self.image_normalize(image)
                    all_cam_images.append(image)


            # get all actions after and including start_ts, with the max length of chunk_size
            action_len = min(episode_len - start_ts, self.chunk_size) 
            action = root['/action'][start_ts:start_ts + action_len]

        # new axis for different cameras
        # image_data = torch.stack(all_cam_images, axis=0)

        # normalize action and qpos
        action = (action - self.mean) / self.std
        qpos = (qpos - self.mean) / self.std

        self.is_sim = is_sim
        padded_action = np.zeros([self.chunk_size, action.shape[1]], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize image and change dtype to float

        # if not self.use_rot:
        #     qpos_data = torch.cat((qpos_data[:3], qpos_data[6:]))
        #     action_data = torch.cat((action_data[:, :3], action_data[:, 6:]), dim=1)

        return all_cam_images, qpos_data, action_data, is_pad

def get_norm_stats(dataset_dir, num_episodes, use_existing=True):
    all_qpos_data = []
    all_action_data = []
    gelsight_means = [] 
    gelsight_stds = []
    use_gelsight = False

    # check to see if norm stats already exists
    if use_existing and os.path.exists(os.path.join(dataset_dir, 'norm_stats.json')):
        with open(os.path.join(dataset_dir, 'norm_stats.json'), 'r') as f:
            stats = json.load(f)
            # convert to numpy array
            for key in stats:
                stats[key] = np.array(stats[key])
            return stats

    for episode_idx in tqdm(range(num_episodes), desc="Get Norm Stats"):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            #qvel = root['/observations/qvel'][()] # unused
            action = root['/action'][()]

            # if gelsight data exists, get the average
            if 'observations/gelsight/depth_strain_image' in root:
                use_gelsight = True
                gelsight_data = root['observations/gelsight/depth_strain_image'][()]
                gelsight_mean = np.mean(gelsight_data, axis=(1, 2))
                gelsight_std = np.std(gelsight_data, axis=(1, 2))   
                gelsight_means.extend(gelsight_mean)
                gelsight_stds.extend(gelsight_std)

        all_qpos_data.append(qpos)
        all_action_data.append(action)
    all_qpos_data = np.concatenate(all_qpos_data, axis=0)
    all_action_data = np.concatenate(all_action_data, axis=0)


    # normalize action data
    action_mean = all_action_data.mean(axis=0)
    action_std = all_action_data.std(axis=0)
    action_std = np.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(axis=0)
    qpos_std = all_qpos_data.std(axis=0)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf) # clipping

    # gelsight mean and std
    if use_gelsight:
        gelsight_mean = np.mean(gelsight_means, axis=0)
        gelsight_std = np.mean(gelsight_stds, axis=0)
        gelsight_std = np.clip(gelsight_std, 1e-2, np.inf)


    stats = {"action_mean": action_mean, "action_std": action_std,
             "qpos_mean": qpos_mean, "qpos_std": qpos_std,
             "example_qpos": qpos}
    
    if use_gelsight:
        stats["gelsight_mean"] = gelsight_mean
        stats["gelsight_std"] = gelsight_std

    return stats


## helper functions, same as ACT

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
