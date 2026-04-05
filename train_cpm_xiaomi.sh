#!/bin/bash
# Train Contact Physical Model (CPM) — Paper-faithful TouchGuide implementation
# 参考 TouchGuide (Table XIII): lr=1e-5, batch=64, epochs=200, warmup=5%
# 前置: 先运行 precompute_dino_xiaomi.sh 提取 DINOv2 特征
#
# 与 TFS 的关键区别:
#   1. 只用当前 (V_t, T_t)，不需要未来触觉
#   2. geometric 噪声采样
#   3. 默认 eef_rel (论文默认 action 表示)
#
# 可选 action_mode: eef_rel(论文默认) / eef_delta / joint_abs

python -m tactile_foresight.training.train_cpm \
  --feature_dir /home/chenshuai/data/dataset/260309_0310_dino_features \
  --hdf5_dir /home/chenshuai/data/dataset/260309_0310 \
  --save_dir /home/chenshuai/Project/output/cpm_xiaomi_eef_rel \
  --num_episodes 337 \
  --start_episode 0 \
  --camera_names global,wrist \
  --action_mode eef_rel \
  --pred_horizon 20 \
  --samples_per_episode 50 \
  --hidden_dim 256 \
  --num_layers 4 \
  --nheads 8 \
  --dropout 0.1 \
  --init_temperature 0.07 \
  --noise_augment \
  --noise_steps 100 \
  --noise_ratio 0.5 \
  --geometric_p 0.05 \
  --epochs 200 \
  --batch_size 64 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --warmup_ratio 0.05 \
  --save_freq 50 \
  --log_freq 10 \
  --plot_freq 50 \
  --num_workers 4 \
  --seed 42
