from load_ACT import load_ACT
import h5py
import json
import numpy as np
from torchvision import transforms
from process_data_cage import MASK_VERTICIES, CROP_PARAMS, make_masks
from typing import Dict, List, Tuple
import cv2
import torch
import os

image_size = (400, 480)

class PreprocessData:
    """Preprocesses the data for the ACT model. Behaves like the dataset class 
    used in the training loop, but does not inherit from torch.utils.data.Dataset."""

    def __init__(self, norm_stats, camera_names):
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.mean = (norm_stats["action_mean"] + norm_stats["qpos_mean"]) / 2
        self.std = (norm_stats["action_std"] + norm_stats["qpos_std"]) / 2
        self.gelsight_mean = norm_stats["gelsight_mean"]
        self.gelsight_std = norm_stats["gelsight_std"]
        self.masks = make_masks(image_size=image_size, verticies=MASK_VERTICIES)
        self.camera_names = camera_names

    def process_data(self, 
                     images: Dict[str, np.ndarray], 
                     gelsight: np.ndarray, 
                     qpos: np.ndarray) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        
        all_images = []
        for cam_name in self.camera_names:

            if cam_name == "gelsight":
                # process gelsight
                gelsight_data = (gelsight - self.gelsight_mean) / self.gelsight_std
                gelsight_data = torch.tensor(gelsight_data, dtype=torch.float32)
                gelsight_data = torch.einsum('h w c -> c h w', gelsight_data)
                all_images.append(gelsight_data.unsqueeze(0))

            elif cam_name in images.keys():
                # crop the images
                crop = CROP_PARAMS[int(cam_name)]

                # crop the image and resize
                image = images[cam_name]
                image = image[crop['i']:crop['i']+crop['h'], crop['j']:crop['j']+crop['w']]
                image = cv2.resize(image, (image_size[1], image_size[0]))

                #apply the masks
                if int(cam_name) in self.masks and self.masks[int(cam_name)] is not None:
                    image = cv2.bitwise_and(image, image, mask=self.masks[int(cam_name)])

                # convert to tensor and normalize
                image = torch.tensor(image, dtype=torch.float32)/255.0
                image = torch.einsum('h w c -> c h w', image) # change to c h w
                image = self.image_normalize(image)
                all_images.append(image.unsqueeze(0))

            else:
                raise ValueError(f"Camera name {cam_name} not found in images")
            
        # get rid of velocities
        qpos = np.concatenate([qpos[:3], qpos[6:]])
        qpos = (qpos - self.mean) / self.std
        qpos_data = torch.from_numpy(qpos).float().unsqueeze(0)

        return all_images, qpos_data        

def visualize(images, qpos, actions, ground_truth):

    import matplotlib.pyplot as plt
    # Create a figure and axes
    fig = plt.figure(figsize=(10, 10), layout='tight')
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    axs_left = subfigs[0].subplots(len(images), 1)
    for i, image in enumerate(images):
        print(image.shape)
        axs_left[i].imshow(image)     

    # Make a 3D scatter plot of the actions in the right subplot. Use cmaps to color the points based on the index
    c = np.arange(len(actions))
    ax2 = subfigs[1].add_subplot(111, projection='3d')
    # ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c='b', marker='o')
    sc = ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c=c, cmap='viridis', marker='x')
    ax2.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], c=c, cmap = 'viridis', marker='o')
    ax2.scatter(qpos[0], qpos[1], qpos[2], c='r', marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Actions and Qpos')
    cbar = fig.colorbar(sc, ax=ax2, label='Time', shrink=0.5)

    # Set the axis limits
    center = np.array([0.5, 0, 0.2])
    radius = 0.15
    ax2.set_xlim(center[0] - radius, center[0] + radius)
    ax2.set_ylim(center[1] - radius, center[1] + radius)
    ax2.set_zlim(center[2] - radius, center[2] + radius)

    plt.show()    
    

if __name__ == '__main__':
    model_path = "/home/aigeorge/research/TactileACT/data/camera_cage/pretrained_1999_melted/policy_last.ckpt"
    args_file = "/home/aigeorge/research/TactileACT/data/camera_cage/pretrained_1999_melted/args.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act = load_ACT(model_path, args_file).to(device)

    gelsight_means = []

    # pretend to be a robot by loading a dataset
    data_dir = "//home/aigeorge/research/TactileACT/data/original/camara_cage_1/run_0/episode_3/episode_3.hdf5"
    with h5py.File(data_dir, 'r') as root:
        qpos = root['/observations/position'][()]
        gt_actions = root['/goal_position'][()]
        all_gelsight_data = root['observations/gelsight/depth_strain_image'][()]
        num_episodes = root.attrs['num_timesteps']

        all_images = {}
        for cam in root.attrs['camera_names']:
            video_images = []
            video_path = os.path.join(os.path.dirname(data_dir), f'cam-{cam}.avi')
            cap = cv2.VideoCapture(video_path)
            for i in range(num_episodes):
                ret, frame = cap.read()
                if not ret:
                    break
                video_images.append(frame)
            
            all_images[cam] = np.array(video_images)
            cap.release()
        

    # get the norm stats from args_file, need to convert to numpy array
    args = json.load(open(args_file, 'r'))
    norm_stats = {k: np.array(v) for k, v in args['norm_stats'].items()}

    # create the fake dataset
    preprocess = PreprocessData(norm_stats, args['camera_names'])

    for i in range(num_episodes):
        images = {key: all_images[key][i] for key in all_images}

        image_data, qpos_data = preprocess.process_data(images, all_gelsight_data[i], qpos[i])
        print(len(image_data))
        print(image_data[0].shape, qpos_data.shape)

        # get the action from the model
        qpos_data = qpos_data.to(device)
        image_data = [img.to(device) for img in image_data]
        all_actions = act(qpos_data, image_data)

        # unnormalize the actions and qpos 
        all_actions = all_actions.squeeze().detach().cpu().numpy() * preprocess.std + preprocess.mean
        qpos_data = qpos_data.squeeze().detach().cpu().numpy() * preprocess.std + preprocess.mean
        print("all actions", all_actions)
        print("qpos", qpos_data.shape)
        print("gt", gt_actions.shape)

        # visualize the data
        if i%10 == 0:
            vis_images = [image_data[j].squeeze().detach().cpu().numpy().transpose(1, 2, 0) for j in range(len(image_data))]
            visualize(vis_images, qpos_data, all_actions, gt_actions[i:i+args['chunk_size']])
