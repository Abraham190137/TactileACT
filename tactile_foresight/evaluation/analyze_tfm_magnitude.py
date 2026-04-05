"""
分析 TFM 预测的变化幅度 vs 真实变化幅度。

核心问题：模型是否只是学了一个"保守的微小偏移"，而没有预测到真实的变化幅度？

对比：
1. 真实的 ||T_{t+k} - T_t|| （真实变化幅度）
2. 预测的 ||T̂_{t+k} - T_t|| （预测变化幅度）
3. 两者的比值（< 1 说明预测幅度不够）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tactile_foresight.models.tfm import TactileForesightModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi_seq20_nomask/tfm_best.pth")
    p.add_argument("--config", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi_seq20_nomask/tfm_config.json")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    pred_horizon = config.get("pred_horizon", 20)
    camera_names = str(config["camera_names"]).split(",")
    feature_dir = config["feature_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TactileForesightModel(
        dino_dim=768, pred_horizon=pred_horizon,
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 4),
        nheads=config.get("nheads", 8),
        dropout=config.get("dropout", 0.1),
        device=device,
    )
    model.load_model(args.checkpoint)
    model.pred_head.eval()

    # 加载验证集
    start = config.get("start_episode", 0)
    num = config["num_episodes"]
    seed = config.get("seed", 42)
    episode_ids = list(range(start, start + num))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    val_ids = [episode_ids[i] for i in shuffled[split:]]

    all_feats = {}
    for eid in val_ids:
        path = os.path.join(feature_dir, "episode_{}_dino.pt".format(eid))
        all_feats[eid] = torch.load(path, map_location="cpu", weights_only=False)

    sample_feat = all_feats[val_ids[0]]
    ep_len = sample_feat["tactile"].shape[0]
    max_t = ep_len - pred_horizon

    # 收集所有样本
    all_v, all_t_cur, all_t_fut = [], [], []
    all_change = []

    rng2 = np.random.RandomState(123)
    for eid in val_ids:
        feats = all_feats[eid]
        for t in range(0, max_t):
            cam = rng2.choice(camera_names)
            v = feats["vision_{}".format(cam)][t].float()
            t_cur = feats["tactile"][t].float()
            t_fut = feats["tactile"][t + 1: t + 1 + pred_horizon].float()
            change = ((t_fut - t_cur.unsqueeze(0)) ** 2).mean().item()
            all_v.append(v)
            all_t_cur.append(t_cur)
            all_t_fut.append(t_fut)
            all_change.append(change)

    all_change = np.array(all_change)
    total = len(all_change)

    # 批量推理
    all_pred = []
    batch_size = 256
    for i in range(0, total, batch_size):
        v_batch = torch.stack(all_v[i:i + batch_size]).to(device)
        t_batch = torch.stack(all_t_cur[i:i + batch_size]).to(device)
        mask = torch.zeros(v_batch.shape[0], dtype=torch.bool, device=device)
        pred = model.predict(v_batch, t_batch, mask)
        all_pred.append(pred.cpu())
    all_pred = torch.cat(all_pred, dim=0)  # (N, H, 768)
    all_t_fut_tensor = torch.stack(all_t_fut)
    all_t_cur_tensor = torch.stack(all_t_cur)

    # 计算变化幅度
    # 真实变化: ||T_{t+k} - T_t||  (L2 norm per step)
    real_delta = all_t_fut_tensor - all_t_cur_tensor.unsqueeze(1)  # (N, H, 768)
    pred_delta = all_pred - all_t_cur_tensor.unsqueeze(1)           # (N, H, 768)

    real_norm = real_delta.norm(dim=-1)  # (N, H)
    pred_norm = pred_delta.norm(dim=-1)  # (N, H)

    # 按变化量分组
    p95 = np.percentile(all_change, 95)
    p50 = np.percentile(all_change, 50)

    groups = [
        ("all", np.ones(total, dtype=bool)),
        ("bottom 50% (static)", all_change <= p50),
        ("top 5% (contact)", all_change > p95),
    ]

    sep = "=" * 100
    print(sep)
    print("  TFM 预测幅度分析：预测的变化量 vs 真实的变化量")
    print(sep)

    for name, mask in groups:
        idx = np.where(mask)[0]
        r_norm = real_norm[idx]  # (n, H)
        p_norm = pred_norm[idx]  # (n, H)

        print("\n  --- {} ({} samples) ---".format(name, len(idx)))
        print("  {:>8} | {:>14} | {:>14} | {:>10} | {:>14}".format(
            "Step", "Real ||delta||", "Pred ||delta||", "Ratio", "说明"))
        print("  " + "-" * 75)

        for step in [0, 4, 9, 14, 19]:
            if step >= pred_horizon:
                continue
            r = r_norm[:, step].mean().item()
            p = p_norm[:, step].mean().item()
            ratio = p / max(r, 1e-8)
            note = ""
            if ratio < 0.5:
                note = "严重不足"
            elif ratio < 0.8:
                note = "偏小"
            elif ratio < 1.2:
                note = "合理"
            else:
                note = "偏大"
            print("  t+{:<4}   | {:>14.4f} | {:>14.4f} | {:>10.2f}x | {:>14}".format(
                step + 1, r, p, ratio, note))

    # 逐步平均 ratio（所有样本）
    print("\n" + sep)
    print("  逐时间步变化幅度比 (pred/real) — 全部样本")
    print(sep)
    ratios_per_step = []
    for step in range(pred_horizon):
        r = real_norm[:, step].mean().item()
        p = pred_norm[:, step].mean().item()
        ratio = p / max(r, 1e-8)
        ratios_per_step.append(ratio)
        print("  t+{:<4}: real={:.4f}  pred={:.4f}  ratio={:.3f}x".format(step + 1, r, p, ratio))

    print("\n  平均 ratio: {:.3f}x".format(np.mean(ratios_per_step)))

    # 额外：看预测是否跟踪了变化方向
    print("\n" + sep)
    print("  变化方向分析：pred_delta 和 real_delta 的 cosine similarity")
    print(sep)
    for name, mask in groups:
        idx = np.where(mask)[0]
        r_d = real_delta[idx]  # (n, H, 768)
        p_d = pred_delta[idx]

        print("\n  --- {} ---".format(name))
        for step in [0, 4, 9, 14, 19]:
            if step >= pred_horizon:
                continue
            cos = F.cosine_similarity(p_d[:, step, :], r_d[:, step, :], dim=-1).mean().item()
            print("  t+{:<4}: delta cosine similarity = {:.4f}".format(step + 1, cos))

    print(sep)


if __name__ == "__main__":
    main()
