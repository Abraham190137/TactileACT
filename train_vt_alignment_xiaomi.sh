#!/bin/bash
# Stage 0: DINOv2 Vision-Tactile Alignment (基于预提取特征)
# 前置: 先运行 precompute_dino_xiaomi.sh 提取 DINOv2 特征
# DINOv2 frozen + projection heads, CLIP contrastive loss

python -m tactile_foresight.training.train_alignment \
  --feature_dir /home/chenshuai/data/dataset/260309_0310_dino_features \
  --save_dir /home/chenshuai/Project/output/vt_align_xiaomi_loop1 \
  --num_episodes 337 \
  --start_episode 0 \
  --camera_names global,wrist \
  --n_clip_images 6 \
  --min_distance 20 \
  --batch_size 128 \
  --shared_dim 256 \
  --hidden_dim 512 \
  --epochs 1000 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --warmup_epochs 20 \
  --save_freq 100 \
  --log_freq 10 \
  --plot_freq 50 \
  --num_workers 4 \
  --seed 42
