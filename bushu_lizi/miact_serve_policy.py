from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import pickle
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_MIACT_ROOT = Path(__file__).resolve().parents[2]
if str(_MIACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIACT_ROOT))

from imitate_episodes import _TactileFrameStack
from imitate_episodes import get_image
from imitate_episodes import get_tactile
from imitate_episodes import make_policy
from imitate_episodes import set_seed
from tools.websocket.miact_ws_server import MIACTWebsocketActionServer
from tools.websocket.miact_ws_server import ClientDisconnected

from config.exp_config import apply_exp_config_to_args
from config.exp_config import load_exp_config


def _build_policy_config(args: argparse.Namespace, *, state_dim: int) -> dict:
    return {
        "lr": float(args.lr),
        "num_queries": int(args.chunk_size),
        "kl_weight": int(args.kl_weight),
        "hidden_dim": int(args.hidden_dim),
        "dim_feedforward": int(args.dim_feedforward),
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "pre_norm": bool(getattr(args, "pre_norm", False)),
        "camera_names": list(args.camera_names),
        "state_dim": int(state_dim),
        # tactile
        "use_tactile": bool(args.use_tactile),
        "tactile_use_depth": bool(args.tactile_use_depth),
        "tactile_use_img": bool(args.tactile_use_img),
        "tactile_use_marker_offset": bool(args.tactile_use_marker_offset),
        "tactile_use_force6d": bool(args.tactile_use_force6d),
        "tactile_use_eef_ft": bool(args.tactile_use_eef_ft),
        "tactile_use_joint_current": bool(args.tactile_use_joint_current),
        "tactile_lowdim_mode": str(getattr(args, "tactile_lowdim_mode", "absrel")),
        "tactile_depth_norm": str(args.tactile_depth_norm),
        "tactile_hands": tuple(args.tactile_hands),
        "tactile_framestack": int(args.tactile_framestack),
        "tactile_vision_grid_size": int(args.tactile_vision_grid_size),
        "tactile_fusion": str(args.tactile_fusion),
        # keep defaults aligned with imitate_episodes
        "tactile_delta": "none",
        "tactile_delta_clip": 0.0,
        "tactile_delta_quantile_prints": 5,
        "use_tactile_residual_inject": False,
        "tactile_residual_scale": 1.0,
        "vision_contact_noise_std": 0.0,
        "anti_freeze_weight": 0.0,
        "use_tactile_gate": False,
        "tactile_gate_hidden": 32,
        "tactile_gate_pool": "last",
        "tactile_gate_source": "force6d",
        "tactile_gate_force_mode": "axis",
        "tactile_gate_force_axis": 2,
        "tactile_gate_loss_weight": 0.0,
        "step9_load_eef_ft_abs_obs": bool(
            getattr(args, "step9_load_eef_ft_abs_obs", False)
        ),
        "ft_framestack": int(getattr(args, "ft_framestack", 1)),
        "step9_load_tactile_force6d_obs": bool(
            getattr(args, "step9_load_tactile_force6d_obs", False)
        ),
        "step9_force_token_inject": str(
            getattr(args, "step9_force_token_inject", "suffix")
        ),
        "step9_load_eef_ft_abs_future": bool(
            getattr(args, "step9_load_eef_ft_abs_future", False)
        ),
        "step9_load_tactile_force6d_future": bool(
            getattr(args, "step9_load_tactile_force6d_future", False)
        ),
        "step9_predict_horizon": int(
            getattr(args, "step9_predict_horizon", int(args.chunk_size))
        ),
        "step9_future_force_loss_weight": float(
            getattr(args, "step9_future_force_loss_weight", 0.0)
        ),
        "step11_future_force_loss_weight": 0.0,
        "step11_tactile_pool": "hand_mean_concat",

        # Step13.2 (phase/event; default off)
        "step13_enabled": bool(getattr(args, "step13_enabled", False)),
        "step13_config": getattr(args, "step13_config", {}),
        "step13_ema_update": getattr(args, "step13_ema_update", None),
        "step13_h_corr": getattr(args, "step13_h_corr", None),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIACT standalone resident websocket policy server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)

    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_best.ckpt")

    parser.add_argument("--policy_class", type=str, default="ACT", choices=["ACT"])
    parser.add_argument("--action_mode", type=str, default="joint", choices=["joint", "eef_rel"])

    parser.add_argument(
        "--exp_config",
        type=str,
        default=None,
        help=(
            "Optional experiment JSON config (Step9/Step13). Must match training to load ckpt. "
            "Example: ./exp_configs/step13/step13_phase_event_noaux_v1.json"
        ),
    )
    parser.add_argument(
        "--pre_norm",
        action="store_true",
        help="Use pre-norm transformer (must match training if enabled)",
    )

    parser.add_argument("--camera_names", nargs="+", default=["global", "wrist"])
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--max_timesteps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1000)

    parser.add_argument("--use_tactile", action="store_true")
    parser.add_argument("--tactile_use_depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tactile_use_img", action="store_true")
    parser.add_argument("--tactile_use_marker_offset", action="store_true")
    parser.add_argument("--tactile_use_force6d", action="store_true")
    parser.add_argument("--tactile_use_eef_ft", action="store_true")
    parser.add_argument("--tactile_use_joint_current", action="store_true")
    parser.add_argument(
        "--tactile_lowdim_mode",
        type=str,
        default="absrel",
        choices=["absrel", "absrel0", "abs"],
        help=(
            "Low-dim tactile token feature mode: absrel=abs+adjacent-diff (default), "
            "absrel0=abs+diff-to-first, abs=abs-only (zero-pad rel)"
        ),
    )
    parser.add_argument("--tactile_depth_norm", type=str, default="amax", choices=["amax", "none"])
    parser.add_argument("--tactile_hands", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--tactile_framestack", type=int, default=1)
    parser.add_argument(
        "--ft_framestack",
        type=int,
        default=1,
        help="Shared: /ft (eef ft abs) history length (default 1)",
    )
    parser.add_argument("--tactile_vision_grid_size", type=int, default=0)
    parser.add_argument("--tactile_fusion", type=str, default="prepend", choices=["prepend", "cross_v2t", "cross_t2v", "cross_bi"])

    parser.add_argument("--log_eval", action="store_true",
                        help="Enable HDF5 trajectory logging for evaluation")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Directory for eval logs "
            "(default: {ckpt_dir}/eval_logs/{RUN_TAG})"
        ),
    )

    return parser


def _update_ft_hist(
    ft_hist: deque[np.ndarray],
    obs: dict,
) -> None:
    ft = obs.get("ft", None)
    if ft is None:
        return
    ft_vec = np.asarray(ft, dtype=np.float32).reshape(-1)
    if ft_vec.shape != (6,):
        raise ValueError(f"Expected obs['ft'] with shape (6,), got {ft_vec.shape}")
    ft_hist.append(ft_vec.copy())


def _normalize_step13_force_source(force_source: str | None) -> str:
    raw = str(force_source or "eef_ft")
    allowed = {"eef_ft", "tac_left_force6d", "joint_current"}
    if raw not in allowed:
        raise ValueError(
            "step13.force_source must be one of "
            "{'eef_ft','tac_left_force6d','joint_current'}, "
            f"got {raw!r}"
        )
    return raw


def _update_step13_hist(
    step13_hist: deque[np.ndarray],
    obs: dict,
    *,
    force_source: str,
) -> None:
    if force_source == "eef_ft":
        ft = obs.get("ft", None)
        if ft is None:
            return
        x = np.asarray(ft, dtype=np.float32).reshape(-1)
        if x.shape != (6,):
            raise ValueError(f"Expected obs['ft'] with shape (6,), got {x.shape}")
        step13_hist.append(x.copy())
        return

    if force_source == "tac_left_force6d":
        tactile = obs.get("tactile", None)
        if not isinstance(tactile, dict):
            return
        left = tactile.get("left", None)
        if not isinstance(left, dict):
            return
        f = left.get("force6d", None)
        if f is None:
            return
        x = np.asarray(f, dtype=np.float32).reshape(-1)
        if x.shape == (12,):
            x = x[:6]
        if x.shape != (6,):
            raise ValueError(
                "Expected obs['tactile']['left']['force6d'] with shape (6,) or (12,), "
                f"got {x.shape}"
            )
        step13_hist.append(x.copy())
        return

    # joint_current
    jc = obs.get("joint_current", None)
    if jc is None:
        return
    x = np.asarray(jc, dtype=np.float32).reshape(-1)
    if x.shape != (7,):
        raise ValueError(f"Expected obs['joint_current'] with shape (7,), got {x.shape}")
    step13_hist.append(x.copy())


def _ft_hist_to_abs_tensor(
    ft_hist: deque[np.ndarray],
    *,
    ft_framestack: int,
    stats: dict,
    device: torch.device,
) -> torch.Tensor:
    T = int(ft_framestack)
    if T < 1:
        raise ValueError(f"ft_framestack must be >= 1, got {T}")
    if len(ft_hist) == 0:
        raise KeyError("Missing obs['ft'] history; got no ft samples")

    frames = list(ft_hist)
    if len(frames) >= T:
        frames = frames[-T:]
    else:
        pad = [frames[0]] * (T - len(frames))
        frames = pad + frames

    ft_np = np.stack(frames, axis=0).astype(np.float32, copy=False)  # (T,6)
    if not np.isfinite(ft_np).all():
        raise ValueError(
            "obs['ft'] contains NaN/Inf. Step13 requires valid 6D wrench. "
            "On robot client, enable publishing ft and ensure the SDK provides force_sensor.zero_force."
        )

    m = stats.get("eef_wrench6d_mean", None)
    s = stats.get("eef_wrench6d_std", None)
    if m is not None and s is not None:
        m = np.asarray(m, dtype=np.float32).reshape(6)
        s = np.asarray(s, dtype=np.float32).reshape(6)
        s = np.maximum(s, 1e-2)
        ft_np = (ft_np - m) / s

    # [B,T,6]
    return torch.from_numpy(ft_np).float().to(device).unsqueeze(0)


def _build_adjacent_diff_np(x_np: np.ndarray) -> np.ndarray:
    if x_np.ndim != 2:
        raise ValueError(f"expected [T,D] array, got {x_np.shape}")
    if x_np.shape[0] <= 0:
        raise ValueError("expected non-empty history for adjacent diff")
    out = np.zeros_like(x_np, dtype=np.float32)
    if x_np.shape[0] >= 2:
        out[1:] = x_np[1:] - x_np[:-1]
    return out


def _step13_source_norm_params(
    stats: dict,
    *,
    force_source: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if force_source == "eef_ft":
        return (
            stats.get("eef_wrench6d_mean", None),
            stats.get("eef_wrench6d_std", None),
            stats.get("eef_wrench6d_rel_adj_mean", None),
            stats.get("eef_wrench6d_rel_adj_std", None),
            stats.get("eef_wrench6d_rel0_mean", None),
            stats.get("eef_wrench6d_rel0_std", None),
        )

    if force_source == "tac_left_force6d":
        mean_dict = stats.get("tactile_force6d_mean", None)
        std_dict = stats.get("tactile_force6d_std", None)
        rel_adj_mean_dict = stats.get("tactile_force6d_rel_adj_mean", None)
        rel_adj_std_dict = stats.get("tactile_force6d_rel_adj_std", None)
        rel0_mean_dict = stats.get("tactile_force6d_rel0_mean", None)
        rel0_std_dict = stats.get("tactile_force6d_rel0_std", None)
        return (
            mean_dict.get("left") if isinstance(mean_dict, dict) else None,
            std_dict.get("left") if isinstance(std_dict, dict) else None,
            rel_adj_mean_dict.get("left") if isinstance(rel_adj_mean_dict, dict) else None,
            rel_adj_std_dict.get("left") if isinstance(rel_adj_std_dict, dict) else None,
            rel0_mean_dict.get("left") if isinstance(rel0_mean_dict, dict) else None,
            rel0_std_dict.get("left") if isinstance(rel0_std_dict, dict) else None,
        )

    return (
        stats.get("joint_current_mean", None),
        stats.get("joint_current_std", None),
        stats.get("joint_current_rel_adj_mean", None),
        stats.get("joint_current_rel_adj_std", None),
        stats.get("joint_current_rel0_mean", None),
        stats.get("joint_current_rel0_std", None),
    )


def _step13_hist_to_force_pack(
    step13_hist: deque[np.ndarray],
    *,
    hist_len: int,
    stats: dict,
    device: torch.device,
    force_source: str,
) -> dict[str, torch.Tensor]:
    T = int(hist_len)
    if T < 1:
        raise ValueError(f"step13 history length must be >= 1, got {T}")
    if len(step13_hist) == 0:
        if force_source == "eef_ft":
            raise KeyError("Missing obs['ft'] history; got no ft samples")
        if force_source == "tac_left_force6d":
            raise KeyError("Missing obs['tactile']['left']['force6d'] history; got no tactile-force samples")
        raise KeyError("Missing obs['joint_current'] history; got no joint_current samples")

    frames = list(step13_hist)
    if len(frames) >= T:
        frames = frames[-T:]
    else:
        pad = [frames[0]] * (T - len(frames))
        frames = pad + frames

    raw_np = np.stack(frames, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(raw_np).all():
        raise ValueError(
            f"Step13 source={force_source} contains NaN/Inf in online obs history"
        )

    m, s, rel_adj_mean, rel_adj_std, rel0_mean, rel0_std = _step13_source_norm_params(
        stats,
        force_source=force_source,
    )

    pack_np: dict[str, np.ndarray] = {
        "abs_raw": raw_np,
        "rel_raw_adj": _build_adjacent_diff_np(raw_np),
        "rel_raw0": raw_np - raw_np[:1],
    }

    abs_np = raw_np.copy()
    abs_std_np = None
    if m is not None and s is not None:
        m_np = np.asarray(m, dtype=np.float32).reshape(-1)
        s_np = np.asarray(s, dtype=np.float32).reshape(-1)
        if m_np.shape[0] != raw_np.shape[1] or s_np.shape[0] != raw_np.shape[1]:
            raise ValueError(
                f"Step13 stats dim mismatch for source={force_source}: "
                f"data_dim={raw_np.shape[1]}, mean_dim={m_np.shape[0]}, std_dim={s_np.shape[0]}"
            )
        s_np = np.maximum(s_np, 1e-2)
        abs_std_np = s_np
        abs_np = (abs_np - m_np) / s_np

    pack_np["abs"] = abs_np
    pack_np["rel"] = _build_adjacent_diff_np(abs_np)
    pack_np["rel0"] = abs_np - abs_np[:1]
    pack_np["absrel"] = np.concatenate([pack_np["abs"], pack_np["rel"]], axis=-1)
    pack_np["absrel0"] = np.concatenate([pack_np["abs"], pack_np["rel0"]], axis=-1)

    if abs_std_np is not None:
        pack_np["rel_adj_norm_absstd"] = pack_np["rel_raw_adj"] / abs_std_np
        pack_np["rel0_norm_absstd"] = pack_np["rel_raw0"] / abs_std_np

    if rel_adj_mean is not None and rel_adj_std is not None:
        rel_adj_mean_np = np.asarray(rel_adj_mean, dtype=np.float32).reshape(-1)
        rel_adj_std_np = np.maximum(np.asarray(rel_adj_std, dtype=np.float32).reshape(-1), 1e-2)
        if rel_adj_mean_np.shape[0] == raw_np.shape[1] and rel_adj_std_np.shape[0] == raw_np.shape[1]:
            pack_np["rel_adj_norm_relstats"] = (
                pack_np["rel_raw_adj"] - rel_adj_mean_np
            ) / rel_adj_std_np

    if rel0_mean is not None and rel0_std is not None:
        rel0_mean_np = np.asarray(rel0_mean, dtype=np.float32).reshape(-1)
        rel0_std_np = np.maximum(np.asarray(rel0_std, dtype=np.float32).reshape(-1), 1e-2)
        if rel0_mean_np.shape[0] == raw_np.shape[1] and rel0_std_np.shape[0] == raw_np.shape[1]:
            pack_np["rel0_norm_relstats"] = (
                pack_np["rel_raw0"] - rel0_mean_np
            ) / rel0_std_np

    return {
        key: torch.from_numpy(val).float().to(device).unsqueeze(0)
        for key, val in pack_np.items()
    }


def _step13_hist_to_abs_tensor(
    step13_hist: deque[np.ndarray],
    *,
    hist_len: int,
    stats: dict,
    device: torch.device,
    force_source: str,
) -> torch.Tensor:
    pack = _step13_hist_to_force_pack(
        step13_hist,
        hist_len=hist_len,
        stats=stats,
        device=device,
        force_source=force_source,
    )
    return pack["abs"]


def main() -> None:
    args = _build_parser().parse_args()

    # Align model construction with imitate_episodes.py: apply exp_config overlays
    # before building policy_config.
    if getattr(args, "exp_config", None):
        exp_path = Path(str(args.exp_config)).expanduser()
        if not exp_path.is_absolute():
            exp_path = (_MIACT_ROOT / exp_path).resolve()
        exp_cfg = load_exp_config(str(exp_path))
        merged = apply_exp_config_to_args(vars(args), exp_cfg)
        for k, v in merged.items():
            setattr(args, k, v)
        args.exp_config = str(exp_path)

    set_seed(int(args.seed))

    step13_cfg = getattr(args, "step13_config", {})
    if step13_cfg is None:
        step13_cfg = {}
    if not isinstance(step13_cfg, dict):
        raise ValueError(f"step13_config must be a dict, got {type(step13_cfg)}")
    step13_force_source = _normalize_step13_force_source(step13_cfg.get("force_source", "eef_ft"))
    step13_event_cfg = step13_cfg.get("event", {}) if isinstance(step13_cfg, dict) else {}
    if not isinstance(step13_event_cfg, dict):
        step13_event_cfg = {}
    step13_residual_cfg = step13_event_cfg.get("residual_corrector", {})
    if not isinstance(step13_residual_cfg, dict):
        step13_residual_cfg = {}
    step13_residual_enabled = str(step13_residual_cfg.get("source", "force")) == "tactile"
    step13_residual_tactile_source = step13_residual_cfg.get("tactile_source", None)
    step13_residual_tactile_framestack = int(
        step13_residual_cfg.get("tactile_framestack", 1) or 1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Missing {stats_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    qpos_dim = int(np.asarray(stats["qpos_mean"]).reshape(-1).shape[0])
    action_dim = int(np.asarray(stats["action_mean"]).reshape(-1).shape[0])

    policy_config = _build_policy_config(args, state_dim=action_dim)
    policy = make_policy(args.policy_class, policy_config)

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    print(policy.load_state_dict(checkpoint))
    policy.to(device)
    policy.eval()
    print(f"Loaded: {ckpt_path}")

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    query_frequency = int(policy_config["num_queries"])
    num_queries = int(policy_config["num_queries"])
    if args.temporal_agg:
        query_frequency = 1

    expected_action_dim = 7 if args.action_mode == "joint" else 6

    tactile_hands = tuple(args.tactile_hands)
    tactile_modalities = []
    if args.tactile_use_depth:
        tactile_modalities.append("depth")
    if args.tactile_use_img:
        tactile_modalities.append("img")
    if args.tactile_use_marker_offset:
        tactile_modalities.append("marker_offset")
    if args.tactile_use_force6d:
        tactile_modalities.append("force6d")

    eval_logger = None
    if args.log_eval:
        from tools.websocket.eval_trajectory_logger import EvalTrajectoryLogger
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = args.log_dir or os.path.join(
            args.ckpt_dir,
            "eval_logs",
            run_tag,
        )
        config_meta = {
            "step13_cfg": getattr(args, "step13_cfg", ""),
            "force_source": step13_force_source,
            "ckpt_path": os.path.join(args.ckpt_dir, args.ckpt_name),
            "exp_config": str(args.exp_config or ""),
            "run_tag": run_tag,
        }
        eval_logger = EvalTrajectoryLogger(log_dir, step13_force_source, config_meta)
        print(f"[miact_serve_policy] eval logging enabled → {log_dir}")

    ws_server = MIACTWebsocketActionServer(host=args.host, port=args.port)
    ws_server.start()
    print(f"[miact_serve_policy] serving on ws://{args.host}:{args.port}")

    try:
        episode_id = 0
        while True:
            try:
                print(f"[miact_serve_policy] waiting first obs for episode {episode_id} ...")
                obs = ws_server.recv_obs()
            except ClientDisconnected:
                print("[miact_serve_policy] client disconnected before first obs; waiting for reconnect")
                continue

            ft_framestack = int(getattr(args, "ft_framestack", 1) or 1)
            step13_hist: deque[np.ndarray] = deque(maxlen=ft_framestack)
            _update_step13_hist(step13_hist, obs, force_source=step13_force_source)

            tactile_stacker = None
            if args.use_tactile and int(args.tactile_framestack) > 1 and len(tactile_modalities) > 0:
                tactile_stacker = _TactileFrameStack(
                    tactile_hands=tactile_hands,
                    tactile_modalities=tuple(tactile_modalities),
                    framestack=int(args.tactile_framestack),
                    default_hw=(128, 128),
                )
            residual_tactile_stacker = None
            if step13_residual_enabled and step13_residual_tactile_framestack > 1:
                residual_tactile_stacker = _TactileFrameStack(
                    tactile_hands=tactile_hands,
                    tactile_modalities=(str(step13_residual_tactile_source),),
                    framestack=step13_residual_tactile_framestack,
                    default_hw=(128, 128),
                )

            if args.temporal_agg:
                all_time_actions = torch.zeros(
                    [int(args.max_timesteps), int(args.max_timesteps) + num_queries, action_dim],
                    device=device,
                )

            if eval_logger is not None:
                eval_logger.begin_episode(episode_id)

            with torch.inference_mode():
                step13_info = {}
                try:
                    for t in range(int(args.max_timesteps)):
                        _update_step13_hist(step13_hist, obs, force_source=step13_force_source)
                        if tactile_stacker is not None:
                            tactile_stacker.update(obs.get("tactile", None))
                        if residual_tactile_stacker is not None:
                            residual_tactile_stacker.update(obs.get("tactile", None))

                        qpos_numpy = np.array(obs["proprio"])
                        qpos = pre_process(qpos_numpy)
                        qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                        if qpos.shape[-1] != qpos_dim:
                            raise ValueError(f"qpos_dim mismatch: expected {qpos_dim}, got {qpos.shape[-1]}")

                        curr_image = get_image(obs, list(args.camera_names), device)

                        # Reset step13_info each step so non-query steps
                        # don't reuse stale P1 data from the previous query.
                        step13_info = None

                        if t % query_frequency == 0:
                            tactile = None
                            if args.use_tactile or bool(getattr(args, "step13_enabled", False)) or step13_residual_enabled:
                                obs_for_tactile = obs
                                obs_for_step13_residual = obs
                                if tactile_stacker is not None:
                                    obs_for_tactile = dict(obs)
                                    obs_for_tactile["tactile"] = tactile_stacker.get_stacked()
                                if residual_tactile_stacker is not None:
                                    obs_for_step13_residual = dict(obs)
                                    obs_for_step13_residual["tactile"] = residual_tactile_stacker.get_stacked()
                                tactile = get_tactile(
                                    obs_for_tactile,
                                    tactile_hands=tactile_hands,
                                    tactile_use_depth=bool(args.tactile_use_depth),
                                    tactile_use_img=bool(args.tactile_use_img),
                                    tactile_use_marker_offset=bool(args.tactile_use_marker_offset),
                                    tactile_use_force6d=bool(args.tactile_use_force6d),
                                    step13_residual_tactile_source=str(step13_residual_tactile_source) if step13_residual_enabled else None,
                                    step13_residual_tactile_obs=obs_for_step13_residual,
                                    device=device,
                                )

                                # Step13 online force history:
                                # - eef_ft:     obs['ft'] -> tactile['eef']['ft']['abs'] + step13_force.abs
                                # - tac_left... obs['tactile']['left']['force6d'] -> step13_force.abs
                                # - joint_current: obs['joint_current'] -> step13_force.abs
                                if bool(getattr(args, "step13_enabled", False)):
                                    step13_force_pack = _step13_hist_to_force_pack(
                                        step13_hist,
                                        hist_len=ft_framestack,
                                        stats=stats,
                                        device=device,
                                        force_source=step13_force_source,
                                    )
                                    eef = tactile.setdefault("eef", {})
                                    if step13_force_source == "eef_ft":
                                        ft_pack = eef.setdefault("ft", {})
                                        ft_pack["abs"] = step13_force_pack["abs"]
                                    step13_pack = eef.setdefault("step13_force", {})
                                    step13_pack.update(step13_force_pack)
                            if eval_logger is not None:
                                all_actions, step13_info = policy(
                                    qpos, curr_image, tactile=tactile,
                                    return_step13_info=True,
                                )
                            else:
                                all_actions = policy(qpos, curr_image, tactile=tactile)
                                step13_info = None

                        if args.temporal_agg:
                            all_time_actions[[t], t:t + num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]

                        raw_action = raw_action.squeeze(0).cpu().numpy()
                        action = post_process(raw_action)
                        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
                        if action_vec.shape[0] != expected_action_dim:
                            raise ValueError(
                                f"Action dim mismatch: expected {expected_action_dim} for action_mode={args.action_mode}, got {action_vec.shape}"
                            )

                        ws_server.send_action(action_vec)
                        if eval_logger is not None:
                            eef_pose = np.asarray(obs.get("eef_pose", []), dtype=np.float32) if "eef_pose" in obs else None
                            eval_logger.record_step(t, obs, action_vec, eef_pose, step13_info=step13_info)
                        if t + 1 < int(args.max_timesteps):
                            obs = ws_server.recv_obs()
                except ClientDisconnected:
                    print("[miact_serve_policy] client disconnected mid-episode; restarting episode")
                    if eval_logger is not None:
                        episode_summary = eval_logger.finish_episode()
                        if episode_summary is not None:
                            print(
                                "[miact_serve_policy] eval episode saved: "
                                f"{episode_summary['path']} "
                                "("
                                f"episode_id={episode_summary['episode_id']}, "
                                f"T_total={episode_summary['T_total']})"
                            )
                    episode_id += 1
                    continue

            if eval_logger is not None:
                episode_summary = eval_logger.finish_episode()
                if episode_summary is not None:
                    print(
                        "[miact_serve_policy] eval episode saved: "
                        f"{episode_summary['path']} "
                        "("
                        f"episode_id={episode_summary['episode_id']}, "
                        f"T_total={episode_summary['T_total']})"
                    )
            episode_id += 1

    except KeyboardInterrupt:
        print("[miact_serve_policy] keyboard interrupt, shutting down")
    finally:
        ws_server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
