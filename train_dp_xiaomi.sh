#!/bin/bash
# Train Diffusion Policy (DP) for xiaomi dataset
# 前置: 先运行 clip_pretrain_xiaomi.sh 训练 CLIP ResNet18 视觉+触觉编码器
#
# 视觉编码器模式 (--obs_encoder):
#   dino_feature:  预提取 DINOv2 特征 (训练快,仅训练用)
#   dino_online:   冻结 DINOv2 在线提取 (训练+推理)
#   resnet18:      随机初始化 ResNet18 端到端训练
#   clip_resnet18: CLIP 预训练 ResNet18 微调 (原版DP方式)
#
# Action 模式 (--action_mode):
#   eef_rel:   相对当前帧 (论文默认, 6D)
#   eef_delta: 帧间差分 (6D)
#   joint_abs: 绝对关节角 (7D)

python -m tactile_foresight.training.train_dp \
  --hdf5_dir /home/chenshuai/data/dataset/260309_0310 \
  --save_dir /home/chenshuai/Project/output/dp_xiaomi_clip \
  --num_episodes 337 \
  --start_episode 0 \
  --camera_names global,wrist \
  --obs_encoder clip_resnet18 \
  --clip_weights_path /home/chenshuai/Project/output/xiaomiloop3/epoch_1999_vision_encoder.pth \
  --no_freeze_vision \
  --share_vision_backbone \
  --use_tactile \
  --clip_tac_weights_path /home/chenshuai/Project/output/xiaomiloop3/epoch_1999_tac_encoder.pth \
  --freeze_tactile \
  --tac_side left \
  --tac_key img \
  --action_mode eef_rel \
  --proprio_key proprio_eef \
  --pred_horizon 20 \
  --samples_per_episode 1 \
  --epochs 1000 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --warmup_steps 500 \
  --num_train_timesteps 100 \
  --num_inference_steps 10 \
  --save_freq 50 \
  --log_freq 10 \
  --plot_freq 50 \
  --num_workers 4 \
  --seed 42
