#!/bin/bash
# TactileACT training for Xiaomi Realman dataset
# Uses separate normalization (NormalizeSeparate), state_dim=7 (joint angles)
#
# Prerequisites:
#   1. CLIP pretraining done (tactal_pretrain_xiaomi.sh)
#   2. meta_data.json created at save_dir/meta_data.json
#   3. data symlinked at save_dir/data/

python imitate_episodes.py \
  --save_dir /home/chenshuai/data/xiaomi_act \
  --name act_joint7_clip_loop1 \
  --policy_class ACT \
  --backbone clip_backbone \
  --vision_backbone_path /home/chenshuai/Project/output/xiaomi/epoch_1099_vision_encoder.pth \
  --gelsight_backbone_path /home/chenshuai/Project/output/xiaomi/epoch_1099_tac_encoder.pth \
  --batch_size 128 \
  --num_epochs 1000 \
  --chunk_size 20 \
  --kl_weight 10 \
  --hidden_dim 512 \
  --dim_feedforward 2048 \
  --lr 4e-5 \
  --seed 1 \
  --temporal_agg \
  --enc_layers 4 \
  --dec_layers 7 \
  --nheads 8 \
  --dropout 0.1 \
  --position_embedding sine \
  --z_dimension 32
