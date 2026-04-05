"""
TFM v2 按触觉变化量分组分析。

核心问题：大部分时间步触觉不变，0.99 cosine sim 被"不变"样本撑起来。
本脚本按未来 20 步的触觉变化量（与当前触觉的 MSE）分组，
单独看"触觉发生变化"时模型的预测质量。

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.evaluation.analyze_tfm_by_change
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tactile_foresight.models.tfm import TactileForesightModel


def parse_args():
    p = argparse.ArgumentParser(description="TFM v2 按触觉变化量分组分析")
    p.add_argument("--checkpoint", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi_seq20/tfm_best.pth")
    p.add_argument("--config", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi_seq20/tfm_config.json")
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def load_val_episodes(config):
    start = config.get("start_episode", 0)
    num = config["num_episodes"]
    seed = config.get("seed", 42)
    episode_ids = list(range(start, start + num))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(len(episode_ids))
    split = int(0.8 * len(episode_ids))
    val_ids = [episode_ids[i] for i in shuffled[split:]]
    return val_ids


def load_features(episode_ids, feature_dir):
    all_feats = {}
    for eid in episode_ids:
        path = os.path.join(feature_dir, f"episode_{eid}_dino.pt")
        all_feats[eid] = torch.load(path, map_location="cpu", weights_only=False)
    return all_feats


@torch.no_grad()
def analyze_by_change(model, all_feats, val_ids, camera_names, pred_horizon, device, batch_size):
    """遍历所有验证集时间步，按触觉变化量分组。"""
    model.pred_head.eval()

    sample_feat = all_feats[val_ids[0]]
    ep_len = sample_feat["tactile"].shape[0]
    max_t = ep_len - pred_horizon

    # 收集所有样本
    all_v, all_t_cur, all_t_fut = [], [], []
    all_change = []  # 未来 20 步与当前触觉的平均 MSE（衡量变化量）

    rng = np.random.RandomState(123)
    for eid in val_ids:
        feats = all_feats[eid]
        # 用所有有效时间步，不采样
        for t in range(0, max_t):
            cam = rng.choice(camera_names)
            v = feats["vision_{}".format(cam)][t].float()
            t_cur = feats["tactile"][t].float()
            t_fut = feats["tactile"][t + 1: t + 1 + pred_horizon].float()

            # 变化量: 未来序列与当前触觉的平均 MSE
            change = ((t_fut - t_cur.unsqueeze(0)) ** 2).mean().item()

            all_v.append(v)
            all_t_cur.append(t_cur)
            all_t_fut.append(t_fut)
            all_change.append(change)

    all_change = np.array(all_change)
    total = len(all_change)

    print("\n[变化量统计]")
    print("  总样本数: {}".format(total))
    print("  变化量 min={:.6f}, max={:.6f}, mean={:.6f}, median={:.6f}".format(
        all_change.min(), all_change.max(), all_change.mean(), np.median(all_change)))

    # 按百分位分组
    percentiles = [0, 50, 75, 90, 95, 100]
    thresholds = np.percentile(all_change, percentiles)
    print("  百分位阈值: {}".format(
        ", ".join(["P{}={:.6f}".format(p, t) for p, t in zip(percentiles, thresholds)])))

    # 批量推理
    all_pred = []
    for i in range(0, total, batch_size):
        v_batch = torch.stack(all_v[i:i + batch_size]).to(device)
        t_batch = torch.stack(all_t_cur[i:i + batch_size]).to(device)
        mask = torch.zeros(v_batch.shape[0], dtype=torch.bool, device=device)
        pred = model.predict(v_batch, t_batch, mask)
        all_pred.append(pred.cpu())
    all_pred = torch.cat(all_pred, dim=0)  # (N, H, 768)
    all_t_fut_tensor = torch.stack(all_t_fut)  # (N, H, 768)
    all_t_cur_tensor = torch.stack(all_t_cur)  # (N, 768)

    # 分组分析
    groups = [
        ("全部样本", np.ones(total, dtype=bool)),
        ("底 50%（几乎不变）", all_change <= thresholds[1]),
        ("50-75%（轻微变化）", (all_change > thresholds[1]) & (all_change <= thresholds[2])),
        ("75-90%（中等变化）", (all_change > thresholds[2]) & (all_change <= thresholds[3])),
        ("90-95%（较大变化）", (all_change > thresholds[3]) & (all_change <= thresholds[4])),
        ("顶 5%（剧烈变化）", all_change > thresholds[4]),
    ]

    sep = "=" * 110
    print("\n" + sep)
    print("  TFM v2 按触觉变化量分组 — 模型 vs Baseline（复制当前触觉）")
    print(sep)
    print("{:<22} | {:>6} | {:>12} | {:>12} | {:>12} | {:>12} | {:>10}".format(
        "分组", "样本数", "模型 CosSim", "模型 MSE", "Base CosSim", "Base MSE", "MSE 降低%"))
    print("-" * 110)

    for name, mask in groups:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        pred = all_pred[idx]          # (n, H, 768)
        target = all_t_fut_tensor[idx]  # (n, H, 768)
        cur = all_t_cur_tensor[idx]    # (n, 768)

        # 模型指标（对所有时间步展开）
        model_cos = F.cosine_similarity(
            pred.reshape(-1, 768), target.reshape(-1, 768), dim=-1).mean().item()
        model_mse = ((pred - target) ** 2).mean().item()

        # Baseline：复制当前触觉
        cur_expand = cur.unsqueeze(1).expand_as(target)  # (n, H, 768)
        base_cos = F.cosine_similarity(
            cur_expand.reshape(-1, 768), target.reshape(-1, 768), dim=-1).mean().item()
        base_mse = ((cur_expand - target) ** 2).mean().item()

        mse_reduce = (base_mse - model_mse) / max(base_mse, 1e-8) * 100

        print("  {:<20} | {:>6d} | {:>12.4f} | {:>12.6f} | {:>12.4f} | {:>12.6f} | {:>+9.1f}%".format(
            name, len(idx), model_cos, model_mse, base_cos, base_mse, mse_reduce))

    # 逐时间步分析（仅顶 5% 变化样本）
    print("\n" + sep)
    print("  顶 5% 剧烈变化样本 — 逐时间步分析")
    print(sep)
    top_idx = np.where(all_change > thresholds[4])[0]
    if len(top_idx) > 0:
        pred_top = all_pred[top_idx]
        target_top = all_t_fut_tensor[top_idx]
        cur_top = all_t_cur_tensor[top_idx]

        print("{:>8} | {:>12} | {:>12} | {:>12} | {:>12} | {:>10}".format(
            "Step", "模型 CosSim", "模型 MSE", "Base CosSim", "Base MSE", "MSE 降低%"))
        print("-" * 80)
        for step in range(pred_horizon):
            p = pred_top[:, step, :]
            t = target_top[:, step, :]
            c = cur_top

            m_cos = F.cosine_similarity(p, t, dim=-1).mean().item()
            m_mse = ((p - t) ** 2).mean().item()
            b_cos = F.cosine_similarity(c, t, dim=-1).mean().item()
            b_mse = ((c - t) ** 2).mean().item()
            reduce = (b_mse - m_mse) / max(b_mse, 1e-8) * 100

            print("  t+{:<4} | {:>12.4f} | {:>12.6f} | {:>12.4f} | {:>12.6f} | {:>+9.1f}%".format(
                step + 1, m_cos, m_mse, b_cos, b_mse, reduce))

    print(sep)


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)
    print("[分析] 配置: {}".format(args.config))

    pred_horizon = config.get("pred_horizon", 20)
    camera_names = str(config["camera_names"]).split(",")
    feature_dir = config["feature_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[分析] device={}".format(device))

    model = TactileForesightModel(
        dino_dim=768, pred_horizon=pred_horizon,
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 4),
        nheads=config.get("nheads", 8),
        dropout=config.get("dropout", 0.1),
        device=device,
    )
    model.load_model(args.checkpoint)
    print("[分析] 加载权重: {}".format(args.checkpoint))

    val_ids = load_val_episodes(config)
    print("[分析] 验证集: {} episodes".format(len(val_ids)))

    t0 = time.time()
    all_feats = load_features(val_ids, feature_dir)
    print("[分析] 特征加载完成 {:.1f}s".format(time.time() - t0))

    analyze_by_change(model, all_feats, val_ids, camera_names, pred_horizon, device, args.batch_size)


if __name__ == "__main__":
    main()
