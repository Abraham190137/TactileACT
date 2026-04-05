#!/bin/bash
# 预提取 DINOv2 特征（一次性，约10-15分钟）
# 输出: 每条轨迹一个 .pt 文件，包含 vision_{cam} (T,768) float16 + tactile (T,768) float16

python -m tactile_foresight.training.precompute_dino_features \
  --dataset_dir /home/chenshuai/data/dataset/260309_0310 \
  --save_dir /home/chenshuai/data/dataset/260309_0310_dino_features \
  --num_episodes 337 \
  --start_episode 0 \
  --camera_names global,wrist \
  --tac_side left \
  --tac_key img \
  --batch_size 64
