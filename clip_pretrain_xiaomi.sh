#!/bin/bash
# CLIP pretraining for xiaomi dataset
# 训练 ResNet18 vision encoder (视觉-触觉对比学习)
# 产出: epoch_*_vision_encoder.pth (用于 train_dp_xiaomi.sh 的 --clip_weights_path)

python clip_pretraining_xiaomi.py \
  --dataset_dir /home/chenshuai/data/dataset/260309_0310 \
  --save_dir /home/chenshuai/Project/output/clip_pretrain_xiaomi \
  --num_episodes 337 \
  --start_episode 0 \
  --camera_names global,wrist \
  --tac_side left \
  --tac_key img \
  --proprio_key proprio_eef \
  --batch_size 3 \
  --n_clip_images 5 \
  --min_distance 20 \
  --n_epochs 200 \
  --clip_dim 512 \
  --resnet_lr 1e-5 \
  --projection_lr 1e-4 \
  --save_freq 25 \
  --plot_freq 50
