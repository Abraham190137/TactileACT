"""
TFM v2 序列预测质量分析脚本。

评估维度：
1. 逐时间步分析 (t+1, t+5, t+10, t+15, t+20)
2. Mask vs No-mask 对比（纯视觉 vs 视觉+触觉）
3. Naive baseline 对比（直接复制当前触觉作为预测）
4. 时间步 × Mask 交叉分析

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.evaluation.analyze_tfm
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
    p = argparse.ArgumentParser(description="TFM v2 序列预测质量分析")
    p.add_argument("--checkpoint", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi_seq20/tfm_best.pth")
    p.add_argument("--config", type=str,
                    default="/home/chenshuai/Project/output/tfm_xiaomi_seq20/tfm_config.json")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--samples_per_episode", type=int, default=50)
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
    train_ids = [episode_ids[i] for i in shuffled[:split]]
    val_ids = [episode_ids[i] for i in shuffled[split:]]
    return train_ids, val_ids


def load_features(episode_ids, feature_dir):
    all_feats = {}
    for eid in episode_ids:
        path = os.path.join(feature_dir, f"episode_{eid}_dino.pt")
        all_feats[eid] = torch.load(path, map_location="cpu", weights_only=False)
    return all_feats


def build_eval_samples(all_feats, episode_ids, camera_names, pred_horizon,
                       samples_per_episode, seed=123):
    """构建评估样本，返回按 masked 分组的列表。

    每个样本: (v_feat, t_cur, t_fut_seq)
    t_fut_seq: (H, 768) 未来 H 步触觉序列
    """
    rng = np.random.RandomState(seed)
    sample_feat = all_feats[episode_ids[0]]
    ep_len = sample_feat["tactile"].shape[0]
    max_t = ep_len - pred_horizon

    samples = {True: [], False: []}  # masked -> list

    for eid in episode_ids:
        feats = all_feats[eid]
        timesteps = rng.randint(0, max_t, size=samples_per_episode)
        cam_choices = rng.choice(camera_names, size=samples_per_episode)

        for t, cam in zip(timesteps, cam_choices):
            v = feats["vision_{}".format(cam)][t].float()
            t_cur = feats["tactile"][t].float()
            t_fut_seq = feats["tactile"][t + 1: t + 1 + pred_horizon].float()  # (H, 768)
            for masked in [True, False]:
                samples[masked].append((v, t_cur, t_fut_seq))

    return samples


@torch.no_grad()
def evaluate_model(model, samples, pred_horizon, device, batch_size=128):
    """评估模型，返回逐时间步 × mask 的指标。"""
    model.pred_head.eval()
    results = {}

    for masked, sample_list in samples.items():
        # 收集所有样本的逐步 cosine sim 和 MSE
        all_step_cos = []   # list of (H,) tensors
        all_step_mse = []
        all_base_cos = []   # baseline: 复制当前触觉
        all_base_mse = []

        for i in range(0, len(sample_list), batch_size):
            batch = sample_list[i:i + batch_size]
            v_feat = torch.stack([s[0] for s in batch]).to(device)
            t_cur = torch.stack([s[1] for s in batch]).to(device)
            t_fut_seq = torch.stack([s[2] for s in batch]).to(device)  # (B, H, 768)
            B = v_feat.shape[0]

            mask_t = torch.full((B,), masked, dtype=torch.bool, device=device)

            # 模型预测: (B, H, 768)
            pred_seq = model.predict(v_feat, t_cur, mask_t)

            # 逐时间步指标
            for step in range(pred_horizon):
                pred_step = pred_seq[:, step, :]      # (B, 768)
                target_step = t_fut_seq[:, step, :]    # (B, 768)

                cos = F.cosine_similarity(pred_step, target_step, dim=-1)  # (B,)
                mse = ((pred_step - target_step) ** 2).mean(dim=-1)        # (B,)

                # baseline: 当前触觉
                base_cos = F.cosine_similarity(t_cur, target_step, dim=-1)
                base_mse = ((t_cur - target_step) ** 2).mean(dim=-1)

                if len(all_step_cos) <= step:
                    all_step_cos.append([])
                    all_step_mse.append([])
                    all_base_cos.append([])
                    all_base_mse.append([])

                all_step_cos[step].append(cos.cpu())
                all_step_mse[step].append(mse.cpu())
                all_base_cos[step].append(base_cos.cpu())
                all_base_mse[step].append(base_mse.cpu())

        # 合并
        for step in range(pred_horizon):
            all_step_cos[step] = torch.cat(all_step_cos[step])
            all_step_mse[step] = torch.cat(all_step_mse[step])
            all_base_cos[step] = torch.cat(all_base_cos[step])
            all_base_mse[step] = torch.cat(all_base_mse[step])

            results[(step, masked)] = {
                "cos_sim_mean": all_step_cos[step].mean().item(),
                "cos_sim_std": all_step_cos[step].std().item(),
                "mse_mean": all_step_mse[step].mean().item(),
                "mse_std": all_step_mse[step].std().item(),
                "baseline_cos_mean": all_base_cos[step].mean().item(),
                "baseline_cos_std": all_base_cos[step].std().item(),
                "baseline_mse_mean": all_base_mse[step].mean().item(),
                "baseline_mse_std": all_base_mse[step].std().item(),
                "count": len(sample_list),
            }

    return results


def print_results(results, pred_horizon):
    sep = "=" * 100

    # 选择展示的时间步
    show_steps = [0, 4, 9, 14, 19]  # t+1, t+5, t+10, t+15, t+20
    show_steps = [s for s in show_steps if s < pred_horizon]

    # --- 1. 关键时间步 × Mask 交叉表 ---
    print("\n" + sep)
    print("  TFM v2 预测质量 — 时间步 × Mask/No-mask 交叉表")
    print(sep)
    print("{:>8} | {:>6} | {:>14} | {:>14} | {:>14} | {:>14}".format(
        "Step", "Mask", "Model CosSim", "Model MSE",
        "Baseline Cos", "Baseline MSE"))
    print("-" * 100)

    for step in show_steps:
        for masked in [False, True]:
            r = results[(step, masked)]
            mask_str = "触觉有" if not masked else "触觉无"
            print(
                "  t+{:<4} | {:>6} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | "
                "{:.4f}±{:.4f} | {:.4f}±{:.4f}".format(
                    step + 1, mask_str,
                    r["cos_sim_mean"], r["cos_sim_std"],
                    r["mse_mean"], r["mse_std"],
                    r["baseline_cos_mean"], r["baseline_cos_std"],
                    r["baseline_mse_mean"], r["baseline_mse_std"],
                ))
        if step != show_steps[-1]:
            print("-" * 100)

    # --- 2. 逐时间步汇总（mask+no-mask 平均）---
    print("\n" + sep)
    print("  逐时间步汇总（mask + no-mask 平均）— 模型 vs Baseline")
    print(sep)
    print("{:>8} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}".format(
        "Step", "Model Cos", "Model MSE", "Base Cos", "Base MSE", "Cos 提升"))
    print("-" * 80)

    for step in range(pred_horizon):
        r0 = results[(step, False)]
        r1 = results[(step, True)]
        avg_cos = (r0["cos_sim_mean"] + r1["cos_sim_mean"]) / 2
        avg_mse = (r0["mse_mean"] + r1["mse_mean"]) / 2
        base_cos = (r0["baseline_cos_mean"] + r1["baseline_cos_mean"]) / 2
        base_mse = (r0["baseline_mse_mean"] + r1["baseline_mse_mean"]) / 2
        delta = avg_cos - base_cos
        print("  t+{:<4} | {:>12.4f} | {:>12.4f} | {:>12.4f} | {:>12.4f} | {:>+12.4f}".format(
            step + 1, avg_cos, avg_mse, base_cos, base_mse, delta))

    # --- 3. Mask vs No-mask 汇总 ---
    print("\n" + sep)
    print("  Mask vs No-mask 汇总（所有时间步平均）")
    print(sep)
    print("{:>16} | {:>12} | {:>12} | {:>8}".format(
        "条件", "CosSim", "MSE", "样本数"))
    print("-" * 55)

    for masked in [False, True]:
        cos_vals = [results[(s, masked)]["cos_sim_mean"] for s in range(pred_horizon)]
        mse_vals = [results[(s, masked)]["mse_mean"] for s in range(pred_horizon)]
        count = results[(0, masked)]["count"]
        label = "纯视觉(masked)" if masked else "视觉+触觉"
        print("  {:>14} | {:>12.4f} | {:>12.4f} | {:>8d}".format(
            label, np.mean(cos_vals), np.mean(mse_vals), count))

    print(sep)


def results_to_serializable(results):
    out = {}
    for (step, masked), v in results.items():
        key = "step{}_{}" .format(step + 1, "masked" if masked else "unmasked")
        out[key] = v
    return out


def main():
    args = parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
        print("[分析] 从配置文件加载: {}".format(args.config))
    else:
        print("[分析] 配置文件不存在，使用默认参数")
        config = {
            "feature_dir": "/home/chenshuai/data/dataset/260309_0310_dino_features",
            "num_episodes": 337, "start_episode": 0,
            "camera_names": "global,wrist", "pred_horizon": 20,
            "hidden_dim": 512, "num_layers": 4, "nheads": 8,
            "dropout": 0.1, "seed": 42,
        }

    pred_horizon = config.get("pred_horizon", 20)
    camera_names = str(config["camera_names"]).split(",")
    feature_dir = config["feature_dir"]

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[分析] device={}".format(device))

    print("[分析] 构建 TFM v2 模型...")
    model = TactileForesightModel(
        dino_dim=768,
        pred_horizon=pred_horizon,
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 4),
        nheads=config.get("nheads", 8),
        dropout=config.get("dropout", 0.1),
        device=device,
    )
    model.load_model(args.checkpoint)
    print("[分析] 加载权重: {}".format(args.checkpoint))
    print("[分析] 可训练参数: {:,}".format(model.num_trainable_params()))

    train_ids, val_ids = load_val_episodes(config)
    print("[分析] 训练集: {} episodes, 验证集: {} episodes".format(len(train_ids), len(val_ids)))

    print("[分析] 加载验证集特征 ({} episodes)...".format(len(val_ids)))
    t0 = time.time()
    all_feats = load_features(val_ids, feature_dir)
    print("[分析] 特征加载完成，耗时 {:.1f}s".format(time.time() - t0))

    print("[分析] 构建评估样本 (每 episode {} 个)...".format(args.samples_per_episode))
    samples = build_eval_samples(
        all_feats, val_ids, camera_names, pred_horizon,
        args.samples_per_episode,
    )
    total = sum(len(v) for v in samples.values())
    print("[分析] 总评估样本数: {}".format(total))

    print("[分析] 开始评估...")
    t0 = time.time()
    results = evaluate_model(model, samples, pred_horizon, device, args.batch_size)
    print("[分析] 评估完成，耗时 {:.1f}s".format(time.time() - t0))

    print_results(results, pred_horizon)

    save_path = os.path.join(output_dir, "tfm_analysis.json")
    with open(save_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "pred_horizon": pred_horizon,
            "num_val_episodes": len(val_ids),
            "samples_per_episode": args.samples_per_episode,
            "results": results_to_serializable(results),
        }, f, indent=2, ensure_ascii=False)
    print("\n[分析] 结果已保存到: {}".format(save_path))


if __name__ == "__main__":
    main()
