

import os
import shutil
import json
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from utils import get_norm_stats

def reorganize_compressed_data(src_dir, save_dir):
    # Reformat hdf5 files collected using the old collection method to a format that 
    # works with the ACT dataloader. (ie, inside of the specified folder, create a 
    # data folder and move all the hdf5 files into it, renaming them to episode_i.hdf5,
    # where i is the episode number starting from 0, and add a metadata.json file)

    # goes through a directory recursivly and copies and renames hdf5 files. 
    # The new names are of the form episode_i.hdf5 where i is the episode number 
    # (starting from 0). The files are copied to a new directory.

    assert os.path.exists(src_dir), "Source directory does not exist"
    assert os.path.exists(save_dir), "Destination directory does not exist"

    # create the data directory
    os.makedirs(os.path.join(save_dir, 'compressed_data'), exist_ok=False)

    # get all files in the source directory (recursively)
    all_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # check if the file is a h5py file
            if file.endswith('.hdf5'):
                all_files.append(os.path.join(root, file))

    # copy and rename the files. Also, rename the dataset "position" to "qpos"
    # and "velocity" to "qvel", and "goal_position" to "action" if they exist:
    for idx, file in enumerate(all_files):
        # copy the file to the new directory
        new_file = os.path.join(save_dir, 'compressed_data', f'episode_{idx}.hdf5')
        shutil.copy(file, new_file)
        # open the file and rename the dataset       
        print(f'Copied and renamed {file} to {new_file}')

    # make the metadata file
    meta_data = {'task_name': 'static_plug',
                'num_episodes': len(all_files),
                'camera_names': ["1", "2", "3", "4", "5"],
                'is_sim': False,
                'state_dim': 4}

    with open(os.path.join(save_dir, 'meta_data.json'), 'w') as f:
        json.dump(meta_data, f)

def uncompress_data(save_dir, use_rot=True):
    # go through all of the hdf5 files in the compressed_data folder, uncompress
    # and process them, and save them in the data folder.

    # First, copy the compressed_data folder to the data folder
    shutil.copytree(os.path.join(save_dir, 'compressed_data'), os.path.join(save_dir, 'data'))

    # get all hdf5 files in the data folder
    all_files = []
    for file in os.listdir(os.path.join(save_dir, 'data')):
        if file.endswith('.hdf5'):
            all_files.append(file)

    # uncompress and process the files
    for file in tqdm(all_files, desc='Uncompressing and processing files'):
        # open the file
        with h5py.File(os.path.join(save_dir, 'data', file), 'a') as hdf5_file:
            image_height = hdf5_file.attrs['image_height']
            image_width = hdf5_file.attrs['image_width']
            # get all entries in the image group:
            for key in hdf5_file['observations/images'].keys():
                # get the image data
                compressed_image_data = hdf5_file['observations/images'][key][()]

                image_data = np.zeros((len(compressed_image_data), image_height, image_width, 3), dtype=np.uint8)
                for i, compressed_image in enumerate(compressed_image_data):
                    # decompress the jpeg image
                    image_data[i] = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

                 # delete the old dataset
                del hdf5_file['observations/images'][key]

                # create a new dataset and save the image data
                hdf5_file.create_dataset(name='observations/images/' + key, 
                                         dtype='uint8', 
                                         chunks=(1, image_height, image_width, 3),
                                         data=image_data)
            
            # rename the datasets to be in the format that the ACT dataloader expects
            if 'observations/position' in hdf5_file:
                hdf5_file.move('observations/position', 'observations/qpos')
            else:
                print(f'No position dataset in file {file}')
            if 'observations/velocity' in hdf5_file:
                hdf5_file.move('observations/velocity', 'observations/qvel')
            else:
                print(f'No velocity dataset in file {file}')
            if '/goal_position' in hdf5_file:
                hdf5_file.move('/goal_position', '/action')

            # # resize gelsight image to match the size of the other images, and move them to the cameras folder
            # if 'observations/gelsight' in hdf5_file:
            #     orginal_images = hdf5_file["observations/gelsight/depth_strain_image"][()]
            #     reshaped_images = np.zeros((len(orginal_images), image_height, image_width, 3), dtype=np.float32)
            #     for i, orginal_image in enumerate(orginal_images):
            #         reshaped_images[i] = cv2.resize(orginal_image, (image_width, image_height))
            #     del hdf5_file["observations/gelsight/depth_strain_image"]
            #     hdf5_file.create_dataset(name="observations/images/gelsight", 
            #                             dtype='float32', 
            #                             chunks=(1, image_height, image_width, 3),
            #                             data=reshaped_images)
                    
            # if the data is not using rotation, removed 3:6 from qpos, action, and qvel
            if not use_rot:
                for key in ['observations/qpos', 'observations/qvel', 'action']:
                    data = hdf5_file[key][()]
                    # print("before delte:", data.shape)
                    data = np.delete(data, [3, 4, 5], axis=1)
                    # print("after delte:", data.shape)
                    del hdf5_file[key]
                    hdf5_file.create_dataset(name=key, data=data, chunks=(1, data.shape[1]))

# def get_gelsight_stats(save_dir):
#     # get all hdf5 files in the data folder
#     all_files = []
#     for file in os.listdir(os.path.join(save_dir, 'compressed_data')):
#         if file.endswith('.hdf5'):
#             all_files.append(file)
    
#     # get the stats
#     means = []
#     stds = []
#     for file in tqdm(all_files):
#         with h5py.File(os.path.join(save_dir, 'compressed_data', file), 'r') as hdf5_file:
#             gelsight_data = hdf5_file['observations/gelsight/depth_strain_image'][()]
#             means.extend(np.mean(gelsight_data, axis=(1, 2)))
#             stds.extend(np.std(gelsight_data, axis=(1, 2)))


#     return np.mean(means, axis=0), np.mean(stds, axis=0)
    
            

unorganized_data_dir = "/home/aigeorge/research/TactileACT/data/1-30-24 Old Lab"
save_dir = "data/old_lab_data"

# reorganize_compressed_data(unorganized_data_dir, save_dir)
# print(get_gelsight_stats(save_dir))
# uncompress_data(save_dir, use_rot=False)

# need to get the number of episodes
num_episodes = 0
while True:
    if os.path.exists(os.path.join(save_dir, 'data', f'episode_{num_episodes}.hdf5')):
        num_episodes += 1
    else:
        break

# get norm stats and save them
stats = get_norm_stats(os.path.join(save_dir, 'data'), num_episodes, use_existing=False)
# convert form numpy to list for json serialization
for key in stats:
    stats[key] = stats[key].tolist()
with open(os.path.join(os.path.join(save_dir, 'data'), 'norm_stats.json'), 'w') as f:
    json.dump(stats, f)

        
    
