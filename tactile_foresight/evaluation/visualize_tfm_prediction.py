"""
TFM v2 预测可视化：挑选触觉变化明显的时刻，画预测 vs 真实触觉轨迹。

1. PCA 降维到 2D，画 20 步预测轨迹 vs 真实轨迹
2. 逐时间步 cosine similarity 曲线
3. 挑选变化最大的 episode 片段

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.evaluation.visualize_tfm_prediction
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

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
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_examples", type=int, default=8)
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


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    pred_horizon = config.get("pred_horizon", 20)
    camera_names = str(config["camera_names"]).split(",")
    feature_dir = config["feature_dir"]
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "vis")
    os.makedirs(output_dir, exist_ok=True)

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
    print("[vis] 加载模型完成")

    val_ids = load_val_episodes(config)
    print("[vis] 验证集: {} episodes".format(len(val_ids)))

    # 找触觉变化最大的 episode 和时间步
    candidates = []  # (change, eid, t)
    for eid in val_ids:
        path = os.path.join(feature_dir, "episode_{}_dino.pt".format(eid))
        feats = torch.load(path, map_location="cpu", weights_only=False)
        tac = feats["tactile"].float()  # (300, 768)
        ep_len = tac.shape[0]
        max_t = ep_len - pred_horizon

        for t in range(0, max_t):
            t_cur = tac[t]
            t_fut = tac[t + 1: t + 1 + pred_horizon]
            change = ((t_fut - t_cur.unsqueeze(0)) ** 2).mean().item()
            candidates.append((change, eid, t, feats))

    # 按变化量排序，取 top
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 从 top 中均匀采样几个（避免都是同一 episode 的连续帧）
    selected = []
    used_eids_t = set()
    for change, eid, t, feats in candidates:
        # 避免同 episode 中太近的时间步
        key = (eid, t // 20)
        if key in used_eids_t:
            continue
        used_eids_t.add(key)
        selected.append((change, eid, t, feats))
        if len(selected) >= args.num_examples:
            break

    print("[vis] 选取 {} 个变化最大的样本".format(len(selected)))

    # ====== 图1: 逐样本 PCA 轨迹 ======
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for idx, (change, eid, t, feats) in enumerate(selected):
        cam = camera_names[0]
        v_feat = feats["vision_{}".format(cam)][t].float().unsqueeze(0).to(device)
        t_cur = feats["tactile"][t].float().unsqueeze(0).to(device)
        t_fut_real = feats["tactile"][t + 1: t + 1 + pred_horizon].float()  # (H, 768)

        mask = torch.zeros(1, dtype=torch.bool, device=device)
        pred = model.predict(v_feat, t_cur, mask).cpu().squeeze(0)  # (H, 768)

        # 收集所有点做 PCA: 当前触觉 + 真实未来 + 预测未来
        t_cur_np = t_cur.cpu().squeeze(0).numpy().reshape(1, -1)
        real_np = t_fut_real.numpy()  # (H, 768)
        pred_np = pred.numpy()        # (H, 768)

        all_points = np.concatenate([t_cur_np, real_np, pred_np], axis=0)  # (1+H+H, 768)
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_points)

        cur_2d = all_2d[0]
        real_2d = all_2d[1:1 + pred_horizon]
        pred_2d = all_2d[1 + pred_horizon:]

        ax = axes[idx]
        # 当前触觉
        ax.scatter(*cur_2d, c="black", s=100, marker="*", zorder=5, label="当前触觉 t")
        # 真实轨迹
        ax.plot(real_2d[:, 0], real_2d[:, 1], "b-o", markersize=3, linewidth=1.5, label="真实", alpha=0.8)
        # 预测轨迹
        ax.plot(pred_2d[:, 0], pred_2d[:, 1], "r--s", markersize=3, linewidth=1.5, label="预测", alpha=0.8)

        # 标注起点终点
        ax.annotate("t+1", real_2d[0], fontsize=7, color="blue")
        ax.annotate("t+20", real_2d[-1], fontsize=7, color="blue")
        ax.annotate("t+1", pred_2d[0], fontsize=7, color="red")
        ax.annotate("t+20", pred_2d[-1], fontsize=7, color="red")

        # 逐步 cosine sim
        cos_per_step = F.cosine_similarity(pred, t_fut_real, dim=-1).numpy()
        avg_cos = cos_per_step.mean()

        ax.set_title("ep{} t={} | change={:.4f}\navg_cos={:.4f}".format(eid, t, change, avg_cos), fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("PC1 ({:.1f}%)".format(pca.explained_variance_ratio_[0] * 100), fontsize=8)
        ax.set_ylabel("PC2 ({:.1f}%)".format(pca.explained_variance_ratio_[1] * 100), fontsize=8)

    fig.suptitle("TFM v2 预测 vs 真实触觉轨迹 (PCA 2D) — 变化最大的样本", fontsize=14)
    fig.tight_layout()
    path1 = os.path.join(output_dir, "tfm_pca_trajectories.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print("[vis] 保存: {}".format(path1))

    # ====== 图2: 逐时间步 cosine similarity 曲线 ======
    fig2, axes2 = plt.subplots(2, 4, figsize=(24, 12))
    axes2 = axes2.flatten()

    for idx, (change, eid, t, feats) in enumerate(selected):
        cam = camera_names[0]
        v_feat = feats["vision_{}".format(cam)][t].float().unsqueeze(0).to(device)
        t_cur = feats["tactile"][t].float().unsqueeze(0).to(device)
        t_fut_real = feats["tactile"][t + 1: t + 1 + pred_horizon].float()

        mask = torch.zeros(1, dtype=torch.bool, device=device)
        pred = model.predict(v_feat, t_cur, mask).cpu().squeeze(0)

        cos_model = F.cosine_similarity(pred, t_fut_real, dim=-1).numpy()
        cos_base = F.cosine_similarity(t_cur.cpu().squeeze(0).unsqueeze(0).expand_as(t_fut_real),
                                        t_fut_real, dim=-1).numpy()
        mse_model = ((pred - t_fut_real) ** 2).mean(dim=-1).numpy()
        mse_base = ((t_cur.cpu().squeeze(0).unsqueeze(0).expand_as(t_fut_real) - t_fut_real) ** 2).mean(dim=-1).numpy()

        ax = axes2[idx]
        steps = np.arange(1, pred_horizon + 1)

        ax.plot(steps, cos_model, "r-o", markersize=3, label="模型 CosSim")
        ax.plot(steps, cos_base, "b--s", markersize=3, label="Baseline CosSim")
        ax.set_ylim(0.93, 1.0)
        ax.set_xlabel("Future step")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("ep{} t={} | change={:.4f}".format(eid, t, change), fontsize=9)
        ax.legend(fontsize=7)

        # 右 y 轴画 MSE
        ax2 = ax.twinx()
        ax2.bar(steps - 0.2, mse_model, width=0.4, alpha=0.3, color="red", label="模型 MSE")
        ax2.bar(steps + 0.2, mse_base, width=0.4, alpha=0.3, color="blue", label="Base MSE")
        ax2.set_ylabel("MSE", fontsize=8)

    fig2.suptitle("TFM v2 逐时间步 CosSim & MSE — 变化最大的样本", fontsize=14)
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "tfm_per_step_metrics.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print("[vis] 保存: {}".format(path2))

    # ====== 图3: 整条 episode 的触觉变化 + 模型预测热力图 ======
    # 挑一个变化最多的 episode
    best_eid = selected[0][1]
    best_feats = selected[0][3]
    tac_all = best_feats["tactile"].float()  # (300, 768)
    ep_len = tac_all.shape[0]
    max_t = ep_len - pred_horizon

    # 计算每个时间步的触觉变化量
    change_per_t = []
    for t in range(1, ep_len):
        c = ((tac_all[t] - tac_all[t - 1]) ** 2).mean().item()
        change_per_t.append(c)
    change_per_t = np.array(change_per_t)

    # 对整条 episode 做预测
    all_preds = []
    cam = camera_names[0]
    for t in range(0, max_t, 1):
        v = best_feats["vision_{}".format(cam)][t].float().unsqueeze(0).to(device)
        tc = tac_all[t].unsqueeze(0).to(device)
        m = torch.zeros(1, dtype=torch.bool, device=device)
        p = model.predict(v, tc, m).cpu().squeeze(0)  # (H, 768)
        all_preds.append(p)

    # 逐步 cosine sim 热力图
    cos_map = np.zeros((max_t, pred_horizon))
    for t in range(max_t):
        real = tac_all[t + 1: t + 1 + pred_horizon]
        pred = all_preds[t]
        cos = F.cosine_similarity(pred, real, dim=-1).numpy()
        cos_map[t] = cos

    fig3, (ax_change, ax_heat) = plt.subplots(2, 1, figsize=(20, 8),
                                               gridspec_kw={"height_ratios": [1, 3]})

    # 上面：触觉变化量曲线
    ax_change.plot(range(1, ep_len), change_per_t, "k-", linewidth=0.8)
    ax_change.fill_between(range(1, ep_len), change_per_t, alpha=0.3)
    ax_change.set_ylabel("触觉变化量\n(帧间 MSE)")
    ax_change.set_title("Episode {} 全轨迹触觉变化 + TFM 预测质量".format(best_eid))
    ax_change.set_xlim(0, ep_len)

    # 下面：cosine sim 热力图
    im = ax_heat.imshow(cos_map.T, aspect="auto", origin="lower",
                         cmap="RdYlGn", vmin=0.95, vmax=1.0,
                         extent=[0, max_t, 0.5, pred_horizon + 0.5])
    ax_heat.set_xlabel("时间步 t")
    ax_heat.set_ylabel("预测步 (t+1 到 t+20)")
    plt.colorbar(im, ax=ax_heat, label="Cosine Similarity")

    fig3.tight_layout()
    path3 = os.path.join(output_dir, "tfm_episode_heatmap.png")
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    print("[vis] 保存: {}".format(path3))

    print("\n[vis] 所有可视化已保存到: {}".format(output_dir))


if __name__ == "__main__":
    main()
