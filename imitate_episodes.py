import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import get_norm_stats, EpisodicDataset, EpisodicDatasetDelta # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy_action_distribution import ACTPolicy
from policy_cnnmlp_distribution import CNNMLPPolicy
from visualize_episodes import save_videos
from visualization_utils import visualize_data, debug

from typing import List, Dict, Tuple, Any
import json

import IPython
e = IPython.embed

import cv2

from visualization_utils import visualise_trajectory, z_slider

DT = 0.01 # need to hardcode, used to be in constants.py

def main(args):

    # extract args
    is_eval:bool = args['eval']
    save_dir:str = args['save_dir']
    model_name:str = args['name']
    policy_class:str = args['policy_class']
    batch_size_train:int = args['batch_size']
    batch_size_val:int = args['batch_size']
    num_epochs:int = args['num_epochs']
    chunk_size:int = args['chunk_size']
    kl_weight:float = args['kl_weight']
    start_kl_epoch:int = args['start_kl_epoch']
    kl_scale_epochs:int = args['kl_scale_epochs']
    hidden_dim:int = args['hidden_dim']
    dim_feedforward:int = args['dim_feedforward']
    lr:float = args['lr']
    lr_backbone:float = args['lr_backbone']
    backbone_type:str = args['backbone']
    num_enc_layers:int = args['enc_layers']
    num_dec_layers:int = args['dec_layers']
    nheads:int = args['nheads']
    seed:int = args['seed']
    temporal_agg:bool = args['temporal_agg']
    ckpt_name:str = args['checkpoint']
    gpu:int = args['gpu']
    onscreen_render:bool = args['onscreen_render'] 
    position_embedding_type:str = args['position_embedding']
    masks:bool = args['masks']
    dilation:bool = args['dilation']
    dropout:float = args['dropout']
    pre_norm:bool = args['pre_norm']
    z_dimension:int = args['z_dimension']
    weight_decay:float = args['weight_decay']


    ckpt_dir: str = os.path.join(save_dir, model_name)
    dataset_dir: str = os.path.join(save_dir, 'data')

    assert os.path.exists(save_dir), f'{save_dir} does not exist. Please select a valid directory.'

    # read the meta_data folder:
    with open(os.path.join(save_dir, 'meta_data.json'), 'r') as f:
        meta_data: Dict[str, Any] = json.load(f)
    task_name: str = meta_data['task_name']
    num_episodes: int = meta_data['num_episodes']
    # episode_len: int = meta_data['episode_length']
    camera_names: List[str] = meta_data['camera_names']
    is_sim: bool = meta_data['is_sim']
    state_dim:int = meta_data['state_dim']

    norm_stats = get_norm_stats(dataset_dir, num_episodes, chunk_size=chunk_size)

    # save norm stats to args. Need to convert from numpy to list
    args['norm_stats'] = {k: v.tolist() for k, v in norm_stats.items()}

    set_seed(1)
    if gpu != -1: # -1 == No GPU Specified
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # load pretrained backbones
    if args['backbone'] == "clip_backbone":
        assert 'gelsight' in camera_names, 'Gelsight camera not found in camera_names. Please add it to the meta_data.json file.'
        from clip_pretraining import modified_resnet18
        gelsight_model = modified_resnet18()
        vision_model = modified_resnet18()
        camera_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
        camera_backbone_mapping['gelsight'] = 1

        # if pretrained, load the weights
        if args['gelsight_backbone_path'] != 'none' and args['vision_backbone_path'] != 'none':
            vision_model.load_state_dict(torch.load(args['vision_backbone_path']))
            gelsight_model.load_state_dict(torch.load(args['gelsight_backbone_path']))
        elif args['gelsight_backbone_path'] != 'none' or args['vision_backbone_path'] != 'none':
            raise ValueError('Both vision and gelsight backbones must be specified if one is specified.')
        
        pretrained_backbones = [vision_model, gelsight_model]
    else:
        if args['gelsight_backbone_path'] != 'none' or args['vision_backbone_path'] != 'none':
            raise ValueError('A backbone path was specified, but the backbone type is not clip_backbone.')
        pretrained_backbones = None
        camera_backbone_mapping = None


    # make policy
    if policy_class == 'ACT':
        policy = ACTPolicy(state_dim=state_dim,
                           hidden_dim=hidden_dim,
                           position_embedding_type=position_embedding_type,
                           lr_backbone=lr_backbone,
                           masks=masks,
                           backbone_type=backbone_type,
                           dilation=dilation,
                           dropout=dropout,
                           nheads=nheads,
                           dim_feedforward=dim_feedforward,
                           num_enc_layers=num_enc_layers,
                           num_dec_layers=num_dec_layers,
                           pre_norm=pre_norm,
                           num_queries=chunk_size,
                           camera_names=camera_names,
                           z_dimension=z_dimension,
                           lr=lr,
                           weight_decay=weight_decay,
                           kl_weight=kl_weight,
                           pretrained_backbones=pretrained_backbones,
                           cam_backbone_mapping=camera_backbone_mapping
                           )

    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(state_dim=state_dim,
                              hidden_dim=hidden_dim,
                              lr_backbone=lr_backbone,
                              masks=masks,
                              backbone_type=backbone_type,
                              dilation=dilation,
                              camera_names=camera_names,
                              lr=lr,
                              weight_decay=weight_decay,
                              )
    else:
        raise NotImplementedError
    
    policy.cuda()
    

    if is_eval:
        import CustomEnv # TODO: change this to the real env
        success_rate, avg_return = eval_bc(policy=policy,
                                           ckpt_dir=ckpt_dir,
                                           ckpt_name=ckpt_name,
                                           policy_class=policy_class,
                                           onscreen_render=onscreen_render,
                                           temporal_agg=temporal_agg,
                                           state_dim=state_dim,
                                           chunk_size=chunk_size,
                                           camera_names=camera_names,
                                           save_episode=True,)
                                        
        print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # If its not eval, then we are training. Make the ckpt_dir to save the model.
    if os.path.exists(ckpt_dir):
        n = 0
        while os.path.exists(ckpt_dir + f'_{n}'):
            n += 1
        Warning(f'{ckpt_dir} already exists. Renaming to {ckpt_dir}_{n}')
        ckpt_dir = ckpt_dir + f'_{n}'

    # make ckpt_dir
    os.makedirs(ckpt_dir)

    # set visualisation directory in debug
    debug.visualizations_dir = os.path.join(ckpt_dir, 'visualizations')

    # save args + meta_data
    combo_dict = {**args, **meta_data}
    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        # use enters for better readability
        json.dump(combo_dict, f, indent=4)

    # load dataset
    # def load_data(config.dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # save dataset stats
    # obtain normalization stats for qpos and action
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)

    # construct dataset and dataloader
    train_dataset = EpisodicDatasetDelta(train_indices, dataset_dir, camera_names, norm_stats, chunk_size=chunk_size)
    val_dataset = EpisodicDatasetDelta(val_indices, dataset_dir, camera_names, norm_stats, chunk_size=chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    debug.action_qpos_normalizer = train_dataset.action_qpos_normalize

    best_ckpt_info = train_bc(policy=policy,
                              train_dataloader=train_dataloader,
                              val_dataloader=val_dataloader,
                              num_epochs=num_epochs,
                              ckpt_dir=ckpt_dir,
                              seed=seed,
                              kl_weight=kl_weight,
                              kl_scale_epochs=kl_scale_epochs,
                              start_kl_epoch=start_kl_epoch,
                              )
                              

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print('checkpoint path', ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def get_image(obs, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(policy:ACTPolicy,
            ckpt_dir:str,
            ckpt_name:str,
            policy_class:str,
            onscreen_render:bool,
            temporal_agg:bool,
            state_dim: int,
            chunk_size: int,
            camera_names,
            save_episode:bool = True
            ):
    
    set_seed(1000)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    # load weights
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    if onscreen_render:
        cv2.namedWindow("plots", cv2.WINDOW_NORMAL)

    # load environment
    env = CustomEnv() # TODO: change this to the real env
    
    env_max_reward = env.MAX_REWARD

    print("temporal_agg:", temporal_agg)

    # query once per chunk 
    query_frequency = chunk_size
    print('query_frequency', query_frequency)
    if temporal_agg:
        query_frequency = 1
        num_queries = chunk_size

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        print(f'Rollout {rollout_id}')
        rollout_id += 0
        ### set task
        obs = env.reset()
        if onscreen_render:
            cv2.imshow("plots", obs['images'][env.CAMERA_NAMES[0]])
            cv2.waitKey(1)

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### process previous timestep to get qpos and image_list
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs, camera_names)

                if onscreen_render:
                    cv2.imshow("plots", obs['images'][env.CAMERA_NAMES[0]])
                    cv2.waitKey(int(DT*1000))

                ### query policy
                if policy_class == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # for i in range(len(actions_for_curr_step)):
                        #     print('action', i, actions_for_curr_step[i])
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                # print('target', target_qpos)
                obs = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(obs['reward'])

            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def train_bc(policy:ACTPolicy,
             train_dataloader: DataLoader,
             val_dataloader: DataLoader, 
             num_epochs: int,
             ckpt_dir: str,
             seed: int,
             kl_weight: float,
             kl_scale_epochs: int,
             start_kl_epoch: int,
             ):
    
    # policy.cuda()
    plot_freq = 50

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                # print("epoch:", epoch, " batch:", batch_idx)
                # print("data 0:", data[0].shape, data[0].dtype)
                # print("data 1:", data[1].shape, data[1].dtype)
                # print("data 2:", data[2].shape, data[2].dtype)
                # print("data 3:", data[3].shape, data[3].dtype)
                # forward pass
                image_data, qpos_data, action_data, is_pad = data
                # visualize_data(image_data, qpos_data, action_data, is_pad)

                # image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
                if batch_idx == 0 and epoch%plot_freq == 0:
                    debug.plot = True
                    debug.print = False
                    debug.epoch = epoch
                    debug.batch = 0
                    debug.dataset = 'validation'

                qpos_data = qpos_data.cuda()
                image_data = [img.cuda() for img in image_data]
                action_data = action_data.cuda()
                is_pad = is_pad.cuda()
                # evaluation, so we want to ignore the latent variables (we still want them to compute the loss though)
                forward_dict = policy(qpos_data, image_data, action_data, is_pad, ignore_latent=True)

                if batch_idx == 0 and epoch%plot_freq == 0:
                    debug.plot = False
                    debug.print = False

                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        policy.optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            # update kl_weight
            if kl_scale_epochs > 0:
                policy.kl_weight = np.clip((epoch - start_kl_epoch) / kl_scale_epochs, 0, 1)*kl_weight
            else:
                policy.kl_weight = np.clip((epoch - start_kl_epoch), 0, 1)*kl_weight

            image_data, qpos_data, action_data, is_pad = data
            if batch_idx == 0 and epoch%plot_freq == 0:
                debug.plot = True
                debug.print = False
                debug.epoch = epoch
                debug.batch = 0
                debug.dataset = 'train'

            qpos_data = qpos_data.cuda()
            image_data = [img.cuda() for img in image_data]
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            forward_dict = policy(qpos_data, image_data, action_data, is_pad)

            if batch_idx == 0 and epoch%plot_freq == 0:
                debug.plot = False
                debug.print = False
            
            # backward
            loss = forward_dict['loss']
            # kl_weight = np.clip((epoch - policy_config['start_kl_epoch']) / policy_config['kl_scale_epochs'], 0, 1)*policy_config['kl_weight']
            # loss = forward_dict['l1'] + forward_dict['kl'] * kl_weight
            loss.backward()
            policy.optimizer.step()
            policy.optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # save history:
    with open(os.path.join(ckpt_dir, 'train_history.pkl'), 'wb') as f:
        pickle.dump(train_history, f)
    with open(os.path.join(ckpt_dir, 'validation_history.pkl'), 'wb') as f:
        pickle.dump(validation_history, f)


    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(1, num_epochs-1, len(train_history)), train_values, label='train') # skip the first epoch
        plt.plot(np.linspace(1, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the JSON config file', default=None)
    args, _ = parser.parse_known_args()

    print('parsed config', args.config)

    # If a config file is provided, use that, overwritting any values specified
    # in the command line
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print('loaded config', config)

        # For the parse to work, we need to specify the type of each argument
        key_types = {
            "eval": bool,
            "checkpoint": str,
            "onscreen_render": bool,
            "save_dir": str,
            "name": str,
            "policy_class": str,
            "batch_size": int,
            "seed": int,
            "num_epochs": int,
            "lr": float,
            "kl_weight": float,
            "start_kl_epoch": int,
            "kl_scale_epochs": int,
            "chunk_size": int,
            "hidden_dim": int,
            "dim_feedforward": int,
            "temporal_agg": bool,
            "z_dimension": int,
            "gpu": int,
            "lr_backbone": float,
            "weight_decay": float,
            "backbone": str,
            "dilation": bool,
            "position_embedding": str,
            "enc_layers": int,
            "dec_layers": int,
            "dropout": float,
            "nheads": int,
            "pre_norm": bool,
            "masks": bool,
            "gelsight_backbone_path": str,
            "vision_backbone_path": str,
        }
        
        for k, v in key_types.items():
            parser.add_argument(f'--{k}', action='store', type=v, help=f'{k}', required=False, default=None)

        # override the default values with the values from the config file
        new_args = vars(parser.parse_args())
        for k, v in new_args.items():
            if v is not None:
                config[k] = v
                print('updated', k, 'to', v)

        print('new config', config)

        # make sure all arguments are present and of the correct type
        for k, v in config.items():
            if k in key_types:
                if not isinstance(v, key_types[k]):
                    raise ValueError(f'Expected {k} to be of type {key_types[k]}, but got {type(v)}')
            elif k != "config":
                raise ValueError(f'Unexpected config parameter {k}')
            
        for k, v in key_types.items():
            if k not in config:
                raise ValueError(f'Expected {k} to be in the config file')

        # check to make sure that non-default values are provided for required arguments:
        if new_args['save_dir'] == "path/to/save_dir":
            raise ValueError('save_dir must be provided in the config file or as a command line argument')
        if new_args['name'] == "example_name":
            raise ValueError('name must be provided in the config file or as a command line argument') 
        if new_args['eval'] == True and new_args['checkpoint'] == "path/to/checkpoint":
            raise ValueError('checkpoint must be provided in the config file or as a command line argument')

            
        # run main with the config
        main(config)
        exit() # exit here, as main(config) will run the training loop

    # If no config file is provided, use the command line arguments  
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', action='store', type=str, help='checkpoint name', required=False, default='policy_best.ckpt')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--save_dir', action='store', type=str, help='Directory where the checkpoint director will be created, where the recoreded sim episodes are saved, and where the meta-data JSON is located', required=True)
    parser.add_argument('--name', action='store', type=str, help='name of the directory (located in save_dir) to store the training data', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=float, help='KL Weight', required=False)
    parser.add_argument('--start_kl_epoch', action='store', type=int, help='start_kl_epoch', required=False, default=0)
    parser.add_argument('--kl_scale_epochs', action='store', type=int, help='number of epochs to scale KL over', required=False, default=0)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help="Size of the embeddings (dimension of the transformer)", required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, default=2048, help="Intermediate size of the feedforward layers in the transformer blocks", required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--z_dimension', default=32, type=int, help='dimension of the latent space of the CVAE')
    # For chosing gpu
    parser.add_argument('--gpu', action='store', type=int, help='chose which gpu to use', required=False, default=-1)

    # from DETR -----------------------------------------------------------------------------------------------------
    parser.add_argument('--lr_backbone', default=1e-5, type=float) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--gelsight_backbone_path', default='none', type=str, required=False)
    parser.add_argument('--vision_backbone_path', default='none', type=str, required=False)

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=7, type=int, 
                        help="Number of decoding layers in the transformer")
    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    
    main(vars(parser.parse_args()))


