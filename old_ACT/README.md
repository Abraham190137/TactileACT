# One ACT Play: Single Demonstration Behavior Cloning with Action Chunking Transformers
This repo is the code for the paper found here: https://arxiv.org/abs/2309.10175

This repo is a fork of ACT (https://github.com/tonyzhaozh/act).

### Repo Structure
- ``my_imitate_episodes.py`` Train and Evaluate ACT
- ``my_policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``my_sim_env.py`` PandaGym based franka test enviroment
- ``my_robot_env.py`` enviroment for running on the real franka robot
- ``constants.py`` Constants shared across files
- ``my_utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation
    conda create -n TactileACT python=3.8
    conda activate TactileACT
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install pexpect
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    pip install tqdm
    cd detr && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate singleDemoACT
    cd <path to act repo>

### Simulated experiments

    python my_record_sim_episodes.py --task_name pick_and_place --save_dir data_local/grip_normalized_test --num_episodes 50 --onscreen_render

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
python my_imitate_episodes.py --task_name pick_and_place --ckpt_dir data_local/my_ckpt_dir --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

