from sre_constants import IN
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

# import IPython
# e = IPython.embed

IS_SIM = True

if IS_SIM:
    from my_sim_env import CustomEnv, AVAILABLE_SIM_NAMES, AVAILABLE_CAM_NAMES
else:
    from my_robot_env import CustomEnv, AVAILABLE_SIM_NAMES
    
from human_policy_generation import HumanLikePolicy

# Constants:
CAMERA_NAMES: List[str] = ['ee', 'isotropic']#['top', 'front', 'side-left']
ACTION_DIM: int = 4
POSITION_DIM: int = 4
# VELOCITY_DIM: int = 3 velocity is not used
# DEMO_NAME: str = "stack.json"
DEMO_NAME: str = "pick_and_place_adjust_edit.json"
# DEMO_NAME: str = "push.json"

INCLUDE_FAILURES: bool = True
TRANSPARENT_ARM: bool = False

print("Include Failures:", INCLUDE_FAILURES)

# Get the path to the SingleDemoFile (parent directory of *this* file), regardless of the working path.
DEMO_FILE: str = os.path.dirname(os.path.abspath(__file__)) + '/' + DEMO_NAME

# Verify that the camera names are valid.
if IS_SIM:
    for cam_name in CAMERA_NAMES:
        assert cam_name in AVAILABLE_CAM_NAMES, f"cam_name: {cam_name} is not a valid camera name. Please choose from: {AVAILABLE_CAM_NAMES}"

def my_main(args):
    failure_numbers = []
    """ Roll out num_episode runs of the 'human-like' policy. Record the images, positions, velocities, and actions."""

    # Parse args.
    TASK_NAME: str = args['task_name'] # To be passed into make_sim_env.
    SAVE_DIR: str = args['save_dir']
    DATASET_DIR: str = os.path.join(SAVE_DIR, 'data')
    NUM_EPISODES: int = args['num_episodes']
    ONSCREEN_RENDER: bool = args['onscreen_render']
    INJECT_NOISE: bool = False
    # RENDER_CAM_NAME: str = 'angle'

    # check to makes sure the specified task is a valid sim task:
    assert TASK_NAME in AVAILABLE_SIM_NAMES, f"TASK_NAME: {TASK_NAME} is not a valid sim task. Please choose from: {AVAILABLE_SIM_NAMES}"

    if IS_SIM:
        env = CustomEnv(task_name=TASK_NAME, inject_noise=INJECT_NOISE, camera_names=CAMERA_NAMES, onscreen_render=ONSCREEN_RENDER, random_start=True, transparent_arm=TRANSPARENT_ARM) # create the environment
    else:
        env = CustomEnv(task_name=TASK_NAME, inject_noise=INJECT_NOISE, random_start=True, estimate_success=False)

    # Create the save directory if it doesn't exist, and save the metadata.
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    meta_data: Dict[str, Any] = {"task_name": TASK_NAME, "num_episodes": NUM_EPISODES, "camera_names": CAMERA_NAMES, "episode_length": env.MAX_TIME_STEPS, "is_sim": True, "transparent_arm": TRANSPARENT_ARM, "include_failures": INCLUDE_FAILURES, 'is_sim': IS_SIM}
    with open(SAVE_DIR + '/meta_data.json', 'w') as f:
        json.dump(meta_data, f)

    # Create the dataset directory if it doesn't exist.
    if not os.path.isdir(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)

    successes: List[int] = [] # 0 for failure, 1 for success
    episode_idx: int = 0
    # policy = HumanLikePolicy(TASK_NAME, DEMO_FILE, env.MAX_TIME_STEPS) # create the policy
    policy = HumanLikePolicy(task_name=TASK_NAME, demo_file=DEMO_FILE, total_steps=env.MAX_TIME_STEPS) # create the policy
    while sum(successes) < NUM_EPISODES: # while we haven't collected enough episodes, keep collecting.
        print(f'{episode_idx=}', 'out of', NUM_EPISODES, 'num_success:', sum(successes))
        episode_idx += 1
        obs: Dict[str, np.ndarray] = env.reset() # reset the environment
        policy.reset(env) # reset the policy
        episode_data: List[Tuple(Dict[str, np.ndarray], np.ndarray)] = []
        for step in range(env.MAX_TIME_STEPS):
            goal_position: np.ndarray = env.normalize_grip(policy(step))
            episode_data.append((deepcopy(obs), deepcopy(goal_position)))
            obs = env.step_normalized_grip(goal_position) # take a step in the environment
            # print('goal_postion', goal_position)
        
        # If it's running on hardware, check if the episode was a success at the end.
        if not IS_SIM:
            while True:
                in_text = input("Was the episode a success? (y/n): ")
                if in_text == 'y':
                    obs['success'] = True
                    break
                elif in_text == 'n':
                    obs['success'] = False
                    break

        if obs['success']:
            successes.append(1)
        else:
            successes.append(0)
            failure_numbers.append(episode_idx-1)
            if not INCLUDE_FAILURES:
                continue # don't save the data if the episode failed.


        # populate the data_dictionary that will be used to save the data.
        data_dict: Dict[str, List[np.ndarray]] = {
            '/observations/qpos': [],
            #'/observations/qvel': [], # velocity is not used
            '/action': [],
        }

        if INCLUDE_FAILURES:
            data_dict['/success'] = [successes[-1]]

        # Add camera entries to the data dict.
        for cam_name in CAMERA_NAMES:
            data_dict[f'/observations/images/{cam_name}'] = []

        for step_data in episode_data:
            data_dict['/observations/qpos'].append(step_data[0]['position'])
            # data_dict['/observations/qvel'].append(step_data[0]['velocity']) # velocity is not used
            data_dict['/action'].append(step_data[1])
            for cam_name in CAMERA_NAMES:
                data_dict[f'/observations/images/{cam_name}'].append(step_data[0]['images'][cam_name])


        # HDF5
        t0: float = time.time()
        if INCLUDE_FAILURES:
            dataset_path: str = os.path.join(DATASET_DIR, f'episode_{episode_idx-1}')
        else:
            dataset_path: str = os.path.join(DATASET_DIR, f'episode_{sum(successes)-1}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs: h5py.Group = root.create_group('observations')
            image: h5py.Group = obs.create_group('images')
            for cam_name in CAMERA_NAMES:
                _ = image.create_dataset(cam_name, (env.MAX_TIME_STEPS, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), 
                                         compression='gzip',  # Use gzip compression
                                         compression_opts=9)  # Set compression level (0-9, where 0 is no compression, 9 is maximum compression))
                
            qpos: h5py.Dataset = obs.create_dataset('qpos', (env.MAX_TIME_STEPS, POSITION_DIM))
            # qvel: h5py.Dataset = obs.create_dataset('qvel', (env.MAX_TIME_STEPS, VELOCITY_DIM)) # velocity is not used
            if INCLUDE_FAILURES:
                success: h5py.Dataset = root.create_dataset(name='success', shape=(1,))
            action: h5py.Dataset = root.create_dataset('action', (env.MAX_TIME_STEPS, ACTION_DIM))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {DATASET_DIR}')
    print(f'Success: {np.sum(successes)} / {len(successes)}')
    print(f'Failures: {failure_numbers}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--save_dir', action='store', type=str, help='Directory to save the data, checkpoints, and metadata in', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    my_main(vars(parser.parse_args()))