import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
from typing import List, Dict, Tuple, Any
import os
import json
from copy import deepcopy
import cv2

import IPython
e = IPython.embed

#INSERT custom env and expert policy here
import CustomEnv
import ExpertPolicy

IS_SIM: bool = True
RENDER_TIME = 0.01

def my_main(args):
    """ Roll out num_episode runs of the 'human-like' policy. Record the images, positions, velocities, and actions."""

    # Parse args.
    task_name: str = args['task_name'] # To be passed into make_sim_env.
    save_dir: str = args['save_dir']
    dataset_dir: str = os.path.join(save_dir, 'data')
    num_episodes: int = args['num_episodes']
    onscreen_render: bool = args['onscreen_render']
    # RENDER_CAM_NAME: str = 'angle'

    env = CustomEnv() #INSERT custom env here

    # Create the save directory if it doesn't exist, and save the metadata.
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    meta_data: Dict[str, Any] = {"task_name": task_name, 
                                 "num_episodes": num_episodes, 
                                 "episode_length": env.MAX_TIME_STEPS, 
                                 'is_sim': IS_SIM,
                                 "camera_names": env.CAMERA_NAMES}
    
    with open(save_dir + '/meta_data.json', 'w') as f:
        json.dump(meta_data, f)

    # Create the dataset directory if it doesn't exist.
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    successes: List[int] = [] # 0 for failure, 1 for success
    episode_idx: int = 0
    # policy = HumanLikePolicy(task_name, DEMO_FILE, env.MAX_TIME_STEPS) # create the policy
    Expert = ExpertPolicy() # create the policy
    while sum(successes) < num_episodes: # while we haven't collected enough episodes, keep collecting.
        print(f'{episode_idx=}', 'out of', num_episodes, 'num_success:', sum(successes))
        episode_idx += 1
        obs: Dict[str, np.ndarray] = env.reset() # reset the environment
        Expert.reset() # reset the policy
        episode_data: List[Tuple(Dict[str, np.ndarray], np.ndarray)] = []
        if onscreen_render:
            cv2.imshow('render', obs['images'][env.CAMERA_NAMES[0]])
            cv2.waitKey(int(RENDER_TIME*1000))
        for step in range(env.MAX_TIME_STEPS):
            goal_qpos: np.ndarray = Expert.get_qpos(obs) # get the action from the policy
            episode_data.append((deepcopy(obs), deepcopy(goal_qpos)))
            obs = env.step(goal_qpos) # take a step in the environment
            if onscreen_render:
                cv2.imshow('render', obs['images'][env.CAMERA_NAMES[0]])
                cv2.waitKey(int(RENDER_TIME*1000))


        if obs['success']:
            successes.append(1)
        else:
            successes.append(0)
            continue # don't save the data if the episode failed.


        # populate the data_dictionary that will be used to save the data.
        data_dict: Dict[str, List[np.ndarray]] = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }

        # Add camera entries to the data dict.
        for cam_name in env.CAMERA_NAMES:
            data_dict[f'/observations/images/{cam_name}'] = []

        for step_data in episode_data:
            data_dict['/observations/qpos'].append(step_data[0]['qpos'])
            data_dict['/observations/qvel'].append(step_data[0]['qvel']) # velocity is not used
            data_dict['/action'].append(step_data[1])
            for cam_name in env.CAMERA_NAMES:
                data_dict[f'/observations/images/{cam_name}'].append(step_data[0]['images'][cam_name])


        # HDF5
        t0 = time.time()
        dataset_path: str = os.path.join(dataset_dir, f'episode_{sum(successes)-1}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            observations: h5py.Group = root.create_group('observations')
            image: h5py.Group = observations.create_group('images')
            for cam_name in env.CAMERA_NAMES:
                _ = image.create_dataset(cam_name, (env.MAX_TIME_STEPS, env.camera_height, env.camera_width, 3), dtype='uint8',
                                         chunks=(1, env.camera_height, env.camera_width, 3), 
                                         compression='gzip',  # Use gzip compression
                                         compression_opts=9)  # Set compression level (0-9, where 0 is no compression, 9 is maximum compression))
                
            qpos: h5py.Dataset = observations.create_dataset('qpos', (env.MAX_TIME_STEPS, env.POSITION_DIM))
            # qvel: h5py.Dataset = obseravition.create_dataset('qvel', (env.MAX_TIME_STEPS, env.VELOCITY_DIM)) # velocity is not used
            action: h5py.Dataset = root.create_dataset('action', (env.MAX_TIME_STEPS, env.ACTION_DIM))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(successes)} / {len(successes)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--save_dir', action='store', type=str, help='Directory to save the data, checkpoints, and metadata in', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    my_main(vars(parser.parse_args()))