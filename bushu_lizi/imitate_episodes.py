import torch
import numpy as np
import os
import sys
import glob
import pickle
import json
import argparse
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import time
import IPython

from config.exp_config import apply_exp_config_to_args, load_exp_config
from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy
from visualize_episodes import save_videos
from tools.cross_attn_train_export import maybe_export_layer0_cross_attn

# realman
from realman_env.envs.realman_env import RealmanEnv, Config as RealmanEnvConfig

e = IPython.embed

# Fixed timestep for saving rollout videos.
# NOTE: constants.py is deprecated; keep DT local to this script.
DT = 0.02


def _tree_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _tree_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_to(v, device) for v in obj)
    return obj


def _load_exp_config(config_path):
    """Compatibility helper for quick CLI checks/debugging."""
    return load_exp_config(config_path)


def _normalize_step13_force_source(force_source):
    raw = str(force_source or "eef_ft")
    allowed = {"eef_ft", "tac_left_force6d", "joint_current"}
    if raw not in allowed:
        raise ValueError(
            "step13.force_source must be one of "
            "{'eef_ft','tac_left_force6d','joint_current'}, "
            f"got {raw!r}"
        )
    return raw


def _resolve_step13_residual_tactile_runtime(step13_config):
    event_cfg = step13_config.get('event', {}) if isinstance(step13_config, dict) else {}
    if not isinstance(event_cfg, dict):
        event_cfg = {}
    residual_cfg = event_cfg.get('residual_corrector', {})
    if not isinstance(residual_cfg, dict):
        residual_cfg = {}
    source = str(residual_cfg.get('source', 'force'))
    tactile_source = residual_cfg.get('tactile_source', None)
    tactile_framestack = int(residual_cfg.get('tactile_framestack', 1) or 1)
    return {
        'enabled': source == 'tactile',
        'source': source,
        'tactile_source': None if tactile_source is None else str(tactile_source),
        'tactile_framestack': tactile_framestack,
    }


def _update_step13_hist_runtime(step13_hist, obs, *, force_source):
    if force_source == 'eef_ft':
        ft = obs.get('ft', None)
        if ft is None:
            return
        x = np.asarray(ft, dtype=np.float32).reshape(-1)
        if x.shape != (6,):
            raise ValueError(f"Expected obs['ft'] with shape (6,), got {x.shape}")
        step13_hist.append(x.copy())
        return

    if force_source == 'tac_left_force6d':
        tactile = obs.get('tactile', None)
        if not isinstance(tactile, dict):
            return
        left = tactile.get('left', None)
        if not isinstance(left, dict):
            return
        f = left.get('force6d', None)
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

    jc = obs.get('joint_current', None)
    if jc is None:
        return
    x = np.asarray(jc, dtype=np.float32).reshape(-1)
    if x.shape != (7,):
        raise ValueError(f"Expected obs['joint_current'] with shape (7,), got {x.shape}")
    step13_hist.append(x.copy())


def _build_adjacent_diff_np(x_np):
    if x_np.ndim != 2:
        raise ValueError(f"expected [T,D] array, got {x_np.shape}")
    if x_np.shape[0] <= 0:
        raise ValueError('expected non-empty history for adjacent diff')
    out = np.zeros_like(x_np, dtype=np.float32)
    if x_np.shape[0] >= 2:
        out[1:] = x_np[1:] - x_np[:-1]
    return out


def _step13_source_norm_params_runtime(stats, *, force_source):
    if force_source == 'eef_ft':
        return (
            stats.get('eef_wrench6d_mean', None),
            stats.get('eef_wrench6d_std', None),
            stats.get('eef_wrench6d_rel_adj_mean', None),
            stats.get('eef_wrench6d_rel_adj_std', None),
            stats.get('eef_wrench6d_rel0_mean', None),
            stats.get('eef_wrench6d_rel0_std', None),
        )
    if force_source == 'tac_left_force6d':
        mean_dict = stats.get('tactile_force6d_mean', None)
        std_dict = stats.get('tactile_force6d_std', None)
        rel_adj_mean_dict = stats.get('tactile_force6d_rel_adj_mean', None)
        rel_adj_std_dict = stats.get('tactile_force6d_rel_adj_std', None)
        rel0_mean_dict = stats.get('tactile_force6d_rel0_mean', None)
        rel0_std_dict = stats.get('tactile_force6d_rel0_std', None)
        return (
            mean_dict.get('left') if isinstance(mean_dict, dict) else None,
            std_dict.get('left') if isinstance(std_dict, dict) else None,
            rel_adj_mean_dict.get('left') if isinstance(rel_adj_mean_dict, dict) else None,
            rel_adj_std_dict.get('left') if isinstance(rel_adj_std_dict, dict) else None,
            rel0_mean_dict.get('left') if isinstance(rel0_mean_dict, dict) else None,
            rel0_std_dict.get('left') if isinstance(rel0_std_dict, dict) else None,
        )
    return (
        stats.get('joint_current_mean', None),
        stats.get('joint_current_std', None),
        stats.get('joint_current_rel_adj_mean', None),
        stats.get('joint_current_rel_adj_std', None),
        stats.get('joint_current_rel0_mean', None),
        stats.get('joint_current_rel0_std', None),
    )


def _step13_hist_to_force_pack_runtime(step13_hist, *, hist_len, stats, device, force_source):
    T = int(hist_len)
    if T < 1:
        raise ValueError(f'step13 history length must be >= 1, got {T}')
    if len(step13_hist) == 0:
        raise KeyError(f'Missing Step13 source history for force_source={force_source}')

    frames = list(step13_hist)
    if len(frames) >= T:
        frames = frames[-T:]
    else:
        frames = [frames[0]] * (T - len(frames)) + frames

    raw_np = np.stack(frames, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(raw_np).all():
        raise ValueError(f'Step13 source={force_source} contains NaN/Inf in online obs history')

    m, s, rel_adj_mean, rel_adj_std, rel0_mean, rel0_std = _step13_source_norm_params_runtime(
        stats,
        force_source=force_source,
    )
    pack_np = {
        'abs_raw': raw_np,
        'rel_raw_adj': _build_adjacent_diff_np(raw_np),
        'rel_raw0': raw_np - raw_np[:1],
    }

    abs_np = raw_np.copy()
    abs_std_np = None
    if m is not None and s is not None:
        m_np = np.asarray(m, dtype=np.float32).reshape(-1)
        s_np = np.maximum(np.asarray(s, dtype=np.float32).reshape(-1), 1e-2)
        if m_np.shape[0] != raw_np.shape[1] or s_np.shape[0] != raw_np.shape[1]:
            raise ValueError(
                f'Step13 stats dim mismatch for source={force_source}: '
                f'data_dim={raw_np.shape[1]}, mean_dim={m_np.shape[0]}, std_dim={s_np.shape[0]}'
            )
        abs_std_np = s_np
        abs_np = (abs_np - m_np) / s_np

    pack_np['abs'] = abs_np
    pack_np['rel'] = _build_adjacent_diff_np(abs_np)
    pack_np['rel0'] = abs_np - abs_np[:1]
    pack_np['absrel'] = np.concatenate([pack_np['abs'], pack_np['rel']], axis=-1)
    pack_np['absrel0'] = np.concatenate([pack_np['abs'], pack_np['rel0']], axis=-1)

    if abs_std_np is not None:
        pack_np['rel_adj_norm_absstd'] = pack_np['rel_raw_adj'] / abs_std_np
        pack_np['rel0_norm_absstd'] = pack_np['rel_raw0'] / abs_std_np

    if rel_adj_mean is not None and rel_adj_std is not None:
        rel_adj_mean_np = np.asarray(rel_adj_mean, dtype=np.float32).reshape(-1)
        rel_adj_std_np = np.maximum(np.asarray(rel_adj_std, dtype=np.float32).reshape(-1), 1e-2)
        if rel_adj_mean_np.shape[0] == raw_np.shape[1] and rel_adj_std_np.shape[0] == raw_np.shape[1]:
            pack_np['rel_adj_norm_relstats'] = (
                pack_np['rel_raw_adj'] - rel_adj_mean_np
            ) / rel_adj_std_np

    if rel0_mean is not None and rel0_std is not None:
        rel0_mean_np = np.asarray(rel0_mean, dtype=np.float32).reshape(-1)
        rel0_std_np = np.maximum(np.asarray(rel0_std, dtype=np.float32).reshape(-1), 1e-2)
        if rel0_mean_np.shape[0] == raw_np.shape[1] and rel0_std_np.shape[0] == raw_np.shape[1]:
            pack_np['rel0_norm_relstats'] = (
                pack_np['rel_raw0'] - rel0_mean_np
            ) / rel0_std_np

    return {
        key: torch.from_numpy(val).float().to(device).unsqueeze(0)
        for key, val in pack_np.items()
    }


def _build_step13_impl_summary(args, step13_config, step13_force_source):
    impl_cfg = step13_config.get('impl', {}) if isinstance(step13_config, dict) else {}
    event_cfg = step13_config.get('event', {}) if isinstance(step13_config, dict) else {}
    event_impl = str(impl_cfg.get('event_impl', 'legacy_scalar_gate'))
    gate_default = 'channel' if event_impl == 'deco_channel_gate' else 'scalar'
    return {
        'compat_mode': str(impl_cfg.get('compat_mode', 'strict_legacy')),
        'phase_impl': str(impl_cfg.get('phase_impl', 'legacy_adaln')),
        'event_impl': event_impl,
        'gate_granularity': str(impl_cfg.get('gate_granularity', gate_default)),
        'h_corr': args.get('step13_h_corr', event_cfg.get('h_corr', None)),
        'force_source': step13_force_source,
    }

def main(args):
    # ########## Step1：Configs ########## #
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    cudnn_benchmark = bool(args.get('cudnn_benchmark', False))
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    if policy_class != 'ACT':
        raise ValueError(f"Unsupported policy_class={policy_class!r}. This repo only supports ACT.")
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    action_mode = args.get('action_mode', 'joint')  # 'joint' or 'eef_rel'
    resume_path = args.get('resume_path', None)
    use_tactile = bool(args.get('use_tactile', False))
    tactile_use_depth = bool(args.get('tactile_use_depth', True))
    tactile_use_img = bool(args.get('tactile_use_img', False))
    tactile_use_marker_offset = bool(args.get('tactile_use_marker_offset', False))
    tactile_use_force6d = bool(args.get('tactile_use_force6d', False))
    tactile_use_eef_ft = bool(args.get('tactile_use_eef_ft', False))
    tactile_use_joint_current = bool(args.get('tactile_use_joint_current', False))
    tactile_lowdim_mode = str(args.get('tactile_lowdim_mode', 'absrel'))
    tactile_depth_norm = str(args.get('tactile_depth_norm', 'amax'))

    # We keep this minimal by convention: only use 'left', 'right', or both ('left','right').
    tactile_hands = tuple(args.get('tactile_hands', ['left', 'right']))
    tactile_framestack = int(args.get('tactile_framestack', 1))
    tactile_vision_grid_size = int(args.get('tactile_vision_grid_size', 0))
    tactile_fusion = str(args.get('tactile_fusion', 'prepend'))
    tactile_delta = str(args.get('tactile_delta', 'none'))
    tactile_delta_clip = float(args.get('tactile_delta_clip', 0.0))
    tactile_delta_quantile_prints = int(args.get('tactile_delta_quantile_prints', 5))
    dataloader_num_workers = int(args.get('dataloader_num_workers', 1))
    dataloader_pin_memory = bool(args.get('dataloader_pin_memory', True))
    dataloader_persistent_workers = bool(args.get('dataloader_persistent_workers', False))
    dataloader_prefetch_factor = int(args.get('dataloader_prefetch_factor', 1))
    use_tactile_residual_inject = bool(args.get('use_tactile_residual_inject', False))
    tactile_residual_scale = float(args.get('tactile_residual_scale', 1.0))
    vision_contact_noise_std = float(args.get('vision_contact_noise_std', 0.0))
    export_cross_attn = bool(args.get('export_cross_attn', False))
    export_cross_attn_interval = int(args.get('export_cross_attn_interval', 0))
    export_cross_attn_dir = args.get('export_cross_attn_dir', None)
    export_cross_attn_layer_idx = args.get('export_cross_attn_layer_idx', [0])
    if export_cross_attn_layer_idx is None:
        export_cross_attn_layer_idx = [0]
    if isinstance(export_cross_attn_layer_idx, int):
        export_cross_attn_layer_idx = [export_cross_attn_layer_idx]
    anti_freeze_weight = float(args.get('anti_freeze_weight', 0.0))

    # cuDNN autotune: may speed up convs when input sizes are stable.
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    # ===== Step-8: tactile contact gate + supervised loss (all default off) =====
    use_tactile_gate = bool(args.get('use_tactile_gate', False))
    tactile_gate_hidden = int(args.get('tactile_gate_hidden', 32))
    tactile_gate_pool = str(args.get('tactile_gate_pool', 'last'))
    tactile_gate_source = str(args.get('tactile_gate_source', 'force6d'))
    tactile_gate_force_mode = str(args.get('tactile_gate_force_mode', 'axis'))
    tactile_gate_force_axis = int(args.get('tactile_gate_force_axis', 2))

    tactile_gate_loss_weight = float(args.get('tactile_gate_loss_weight', 0.0))
    tactile_gate_return_meta = bool(args.get('tactile_gate_return_meta', False))
    tactile_gate_key = str(args.get('tactile_gate_key', '/t_onset/stable_global'))
    tactile_gate_load_force6d = bool(args.get('tactile_gate_load_force6d', False))
    tactile_gate_load_marker_offset = bool(args.get('tactile_gate_load_marker_offset', False))

    # ===== Step-8 diag: gate observability (default off) =====
    gate_diag_interval = int(args.get('gate_diag_interval', 0))
    gate_diag_max_batches = int(args.get('gate_diag_max_batches', 5))
    gate_diag_out_dir = args.get('gate_diag_out_dir', None)

    # ===== Step-9 (Phase A): dataset targets (all default off) =====
    step9_load_eef_ft_abs_obs = bool(args.get('step9_load_eef_ft_abs_obs', False))
    if 'step9_eef_wrench_framestack' in args:
        raise ValueError(
            "Deprecated: step9_eef_wrench_framestack has been removed. "
            "Use the shared --ft_framestack knob instead."
        )
    step9_load_tactile_force6d_obs = bool(args.get('step9_load_tactile_force6d_obs', False))
    if 'step9_tactile_force6d_framestack' in args:
        raise ValueError(
            "Deprecated: step9_tactile_force6d_framestack has been removed. "
            "Use tactile_framestack as the shared history length for per-hand force6d."
        )
    step9_force_token_inject = str(args.get('step9_force_token_inject', 'suffix'))
    if step9_force_token_inject not in ('prefix', 'suffix'):
        raise ValueError(
            "--step9_force_token_inject must be 'prefix' or 'suffix', "
            f"got {step9_force_token_inject!r}"
        )
    step9_load_eef_ft_abs_future = bool(args.get('step9_load_eef_ft_abs_future', False))
    step9_load_tactile_force6d_future = bool(args.get('step9_load_tactile_force6d_future', False))
    step9_future_force_loss_weight = float(args.get('step9_future_force_loss_weight', 0.0))

    # ===== Step11.1: tactile_latent_shared -> future force prediction (default off) =====
    step11_future_force_loss_weight = float(args.get('step11_future_force_loss_weight', 0.0))
    step11_tactile_pool = str(args.get('step11_tactile_pool', 'hand_mean_concat'))
    step9_enable_targets = bool(
        step9_load_eef_ft_abs_obs
        or step9_load_tactile_force6d_obs
        or step9_load_eef_ft_abs_future
        or step9_load_tactile_force6d_future
    )
    # Step9 未来窗口长度：默认对齐 chunk_size，但允许单独覆盖。
    step9_predict_horizon = int(args.get('step9_predict_horizon', 0) or 0)
    if step9_predict_horizon <= 0:
        step9_predict_horizon = int(args.get('chunk_size', 0) or 0)
    if (step9_load_eef_ft_abs_future or step9_load_tactile_force6d_future) and step9_predict_horizon <= 0:
        raise ValueError(
            "Step9 future targets enabled but horizon is invalid. "
            "Set --step9_predict_horizon or --chunk_size to a positive int."
        )

    ft_framestack = int(args.get('ft_framestack', 1) or 0)
    if ft_framestack <= 0:
        raise ValueError(f"ft_framestack must be >= 1, got {ft_framestack}")

    if step9_future_force_loss_weight < 0:
        raise ValueError(
            f"step9_future_force_loss_weight must be >= 0, got {step9_future_force_loss_weight}"
        )
    if step9_future_force_loss_weight > 0 and not (
        step9_load_eef_ft_abs_future or step9_load_tactile_force6d_future
    ):
        raise ValueError(
            "step9_future_force_loss_weight>0 requires future targets; "
            "enable --step9_load_eef_ft_abs_future and/or --step9_load_tactile_force6d_future"
        )

    if step11_future_force_loss_weight < 0:
        raise ValueError(
            f"step11_future_force_loss_weight must be >= 0, got {step11_future_force_loss_weight}"
        )
    if step11_tactile_pool not in ('hand_mean_concat', 'all_mean'):
        raise ValueError(
            "--step11_tactile_pool must be 'hand_mean_concat' or 'all_mean', "
            f"got {step11_tactile_pool!r}"
        )
    if step11_future_force_loss_weight > 0 and not use_tactile:
        raise ValueError(
            "step11_future_force_loss_weight>0 requires --use_tactile"
        )
    if step11_future_force_loss_weight > 0 and step9_future_force_loss_weight > 0:
        raise ValueError(
            "Do not enable both Step9 and Step11 future-force aux loss in the same run. "
            "Set either --step9_future_force_loss_weight or --step11_future_force_loss_weight."
        )
    if step11_future_force_loss_weight > 0 and not (
        step9_load_eef_ft_abs_future or step9_load_tactile_force6d_future
    ):
        raise ValueError(
            "step11_future_force_loss_weight>0 requires future targets; "
            "enable --step9_load_eef_ft_abs_future and/or --step9_load_tactile_force6d_future"
        )

    # ===== Step13.1: dataset-provided eef wrench history (data-only; default off) =====
    step13_enabled = bool(args.get('step13_enabled', False))
    step13_config = args.get('step13_config', {})
    if step13_config is None:
        step13_config = {}
    if not isinstance(step13_config, dict):
        raise ValueError(
            f"step13_config must be a dict when provided, got {type(step13_config)}"
        )
    if 'Tw' in step13_config:
        raise ValueError(
            "Deprecated: step13.Tw has been removed. Use the shared --ft_framestack knob instead."
        )
    step13_force_source = _normalize_step13_force_source(step13_config.get('force_source', 'eef_ft'))
    step13_config['force_source'] = step13_force_source
    step13_enable_eef_wrench_hist = bool(
        args.get('step13_enable_eef_wrench_hist', step13_enabled)
    )
    step13_aux_cfg = step13_config.get('aux', {}) if isinstance(step13_config, dict) else {}
    if not isinstance(step13_aux_cfg, dict):
        raise ValueError(
            f"step13.aux must be a dict when provided, got {type(step13_aux_cfg)}"
        )
    step13_aux_enabled = bool(step13_enabled and step13_aux_cfg.get('enabled', False))
    step13_aux_future_dw_loss_weight = float(step13_aux_cfg.get('weight', 0.0) or 0.0)
    if step13_aux_future_dw_loss_weight < 0:
        raise ValueError(
            "step13.aux.weight must be >= 0, "
            f"got {step13_aux_future_dw_loss_weight}"
        )
    if step13_aux_future_dw_loss_weight > 0 and not step13_aux_enabled:
        raise ValueError(
            "step13.aux.weight>0 requires step13.aux.enabled=true"
        )
    # Step13.4 约定：目标定义与归一化空间跟随 event 三轴配置。
    step13_event_cfg = step13_config.get('event', {}) if isinstance(step13_config, dict) else {}
    if not isinstance(step13_event_cfg, dict):
        raise ValueError(
            f"step13.event must be a dict when provided, got {type(step13_event_cfg)}"
        )
    step13_residual_cfg = step13_event_cfg.get('residual_corrector', {})
    if not isinstance(step13_residual_cfg, dict):
        raise ValueError(
            "step13.event.residual_corrector must be a dict when provided, "
            f"got {type(step13_residual_cfg)}"
        )
    step13_residual_source = str(step13_residual_cfg.get('source', 'force'))
    step13_residual_tactile_source = step13_residual_cfg.get('tactile_source', None)
    step13_residual_tactile_framestack = int(
        step13_residual_cfg.get('tactile_framestack', 1) or 1
    )
    step13_aux_dw_mode = str(step13_event_cfg.get('dw_mode', 'adjacent'))
    step13_aux_dw_space = str(step13_event_cfg.get('dw_space', 'norm_then_diff'))
    step13_aux_dw_norm_stats = str(step13_event_cfg.get('dw_norm_stats', 'abs_std'))
    if step13_aux_dw_mode not in ('adjacent', 'to_first'):
        raise ValueError(
            "step13 aux expects event.dw_mode in {'adjacent','to_first'}, "
            f"got {step13_aux_dw_mode!r}"
        )
    if step13_aux_dw_space not in ('norm_then_diff', 'raw_then_norm'):
        raise ValueError(
            "step13 aux expects event.dw_space in {'norm_then_diff','raw_then_norm'}, "
            f"got {step13_aux_dw_space!r}"
        )
    if step13_aux_dw_norm_stats not in ('abs_std', 'rel_stats'):
        raise ValueError(
            "step13 aux expects event.dw_norm_stats in {'abs_std','rel_stats'}, "
            f"got {step13_aux_dw_norm_stats!r}"
        )
    # Step13.4 默认未来窗口长度对齐 chunk_size。
    step13_aux_predict_horizon = int(args.get('chunk_size', 0) or 0)
    if step13_aux_enabled and step13_aux_predict_horizon <= 0:
        raise ValueError(
            "step13 aux enabled but chunk_size/horizon is invalid. "
            "Set --chunk_size to a positive int."
        )
    if step13_aux_enabled:
        # Step13.4 targets rely on step13 force history pack in dataset.
        step13_enable_eef_wrench_hist = True
    if step13_enabled:
        # Step13.2/13.3 also require force history (even without aux).
        step13_enable_eef_wrench_hist = True
    if 'step13_eef_wrench_hist_Tw' in args:
        raise ValueError(
            "Deprecated: step13_eef_wrench_hist_Tw has been removed. "
            "Use the shared --ft_framestack knob instead."
        )
    if step13_enable_eef_wrench_hist and ft_framestack <= 0:
        raise ValueError(
            "Step13 eef wrench hist enabled but ft_framestack is invalid. "
            "Set --ft_framestack to a positive int."
        )

    if step9_enable_targets and (step13_enabled or step13_enable_eef_wrench_hist):
        raise ValueError(
            "Step9 and Step13 are mutually exclusive; do not enable both in the same run."
        )

    if bool(args.get('tactile_self_test', False)):
        _self_test_tactile_framestack()
        return

    if tactile_fusion not in ("prepend", "cross_v2t", "cross_t2v", "cross_bi"):
        raise ValueError(
            f"Unsupported tactile_fusion={tactile_fusion!r}. Expected 'prepend', 'cross_v2t', 'cross_t2v', or 'cross_bi'."
        )

    if tactile_delta not in ("none", "adjacent"):
        raise ValueError(
            f"Unsupported tactile_delta={tactile_delta!r}. Expected 'none' or 'adjacent'."
        )
    if tactile_delta_clip < 0:
        raise ValueError(f"tactile_delta_clip must be >= 0, got {tactile_delta_clip}")

    if tactile_delta_quantile_prints < 0:
        raise ValueError(
            "tactile_delta_quantile_prints must be >= 0, "
            f"got {tactile_delta_quantile_prints}"
        )

    if dataloader_num_workers < 0:
        raise ValueError(
            f"dataloader_num_workers must be >= 0, got {dataloader_num_workers}"
        )
    if dataloader_prefetch_factor < 1:
        raise ValueError(
            f"dataloader_prefetch_factor must be >= 1, got {dataloader_prefetch_factor}"
        )
    if dataloader_persistent_workers and dataloader_num_workers == 0:
        raise ValueError(
            "--dataloader_persistent_workers requires --dataloader_num_workers > 0"
        )

    if tactile_residual_scale < 0:
        raise ValueError(
            f"tactile_residual_scale must be >= 0, got {tactile_residual_scale}"
        )
    if vision_contact_noise_std < 0:
        raise ValueError(
            f"vision_contact_noise_std must be >= 0, got {vision_contact_noise_std}"
        )
    if (use_tactile_residual_inject or vision_contact_noise_std > 0) and not use_tactile:
        raise ValueError(
            "Step10.7 requires --use_tactile (to get tactile features / gate meta)."
        )

    if anti_freeze_weight < 0:
        raise ValueError(f"anti_freeze_weight must be >= 0, got {anti_freeze_weight}")

    # tactile_vision_grid_size semantics (img tokens only):
    # - 0: native feature map tokens (h*w per hand)
    # - -1: global token per hand (GAP)
    # - N>0: force NxN grid tokens per hand
    if tactile_vision_grid_size == 0:
        tactile_vision_grid_size = None
    elif tactile_vision_grid_size < -1:
        raise ValueError(f"tactile_vision_grid_size must be -1, 0, or >0, got {tactile_vision_grid_size}")

    # ===== tactile 工程语义（总开关 + 模态选择） =====
    # - use_tactile=False: 不启用主干 tactile 路径。
    #   例外：若 step13.event.residual_corrector.source=tactile，dataset 仍会返回
    #   tactile['step13_residual'] 供 Step13 residual 专用分支使用。
    # - use_tactile=True: 由 tactile_use_* 选择具体模态

    # get task parameters
    dataset_dir = str(args['dataset_dir'])
    num_episodes = int(args['num_episodes'])
    episode_len = int(args['episode_len'])
    camera_names = list(args['camera_names'])

    step13_impl_summary = _build_step13_impl_summary(args, step13_config, step13_force_source)
    print(f"step13_impl_summary={json.dumps(step13_impl_summary, ensure_ascii=False, separators=(',', ':'))}")

    # ===== Step2：动态获取观测维度(qpos)和动作维度 =====
    # 训练：从 dataset 的 episode_0.hdf5 读取
    # 部署/评估(--eval)：不强依赖 dataset，直接从 ckpt_dir/dataset_stats.pkl 推断
    if is_eval:
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"Missing {stats_path}. Eval mode requires dataset_stats.pkl in ckpt_dir."
            )
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        qpos_dim = int(np.asarray(stats['qpos_mean']).reshape(-1).shape[0])
        action_dim = int(np.asarray(stats['action_mean']).reshape(-1).shape[0])
    else:
        import h5py

        first_episode_path = os.path.join(dataset_dir, "episode_0.hdf5")
        with h5py.File(first_episode_path, 'r') as f:
            # 观测维度：关节维度（例如 7）
            qpos_dim = f['/observations/proprio_joint'].shape[1]
            # 动作维度：根据 action_mode 选择 joint_abs 或 eef_rel
            if action_mode == 'eef_rel':
                action_dim = f['/actions/eef_rel'].shape[1]
            else:
                action_dim = f['/actions/joint_abs'].shape[1]

    # 对于 ACT 模型，"state_dim" 实际上用作动作维度（decoder 输出 / encoder action 输入）
    state_dim = action_dim
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        pre_norm = bool(args.get('pre_norm', False))
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'pre_norm': pre_norm,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         # tactile (model-side; default off)
                         'use_tactile': use_tactile,
                         'tactile_use_depth': tactile_use_depth,
                         'tactile_use_img': tactile_use_img,
                         'tactile_use_marker_offset': tactile_use_marker_offset,
                         'tactile_use_force6d': tactile_use_force6d,
                         'tactile_use_eef_ft': tactile_use_eef_ft,
                         'tactile_use_joint_current': tactile_use_joint_current,
                         'tactile_lowdim_mode': tactile_lowdim_mode,
                         # Step13.2 (phase/event; default off). These keys are
                         # consumed by detr/models/detr_vae.py via getattr(args,...)
                         # after build_ACT_model_and_optimizer constructs an argparse args.
                         'step13_enabled': step13_enabled,
                         'step13_config': step13_config,
                         'step13_ema_update': args.get('step13_ema_update', None),
                         'step13_h_corr': args.get('step13_h_corr', None),
                         'tactile_depth_norm': tactile_depth_norm,
                         'tactile_hands': tactile_hands,
                         'tactile_framestack': tactile_framestack,
                         'tactile_vision_grid_size': tactile_vision_grid_size,
                         'tactile_fusion': tactile_fusion,
                         # Step-7 (default off)
                         'tactile_delta': tactile_delta,
                         'tactile_delta_clip': tactile_delta_clip,
                         'tactile_delta_quantile_prints': tactile_delta_quantile_prints,

                         # Step10.7 (default off)
                         'use_tactile_residual_inject': use_tactile_residual_inject,
                         'tactile_residual_scale': tactile_residual_scale,
                         'vision_contact_noise_std': vision_contact_noise_std,
                         # Step-7.3 (default off)
                         'anti_freeze_weight': anti_freeze_weight,

                         # Step-8.2: gate module knobs (default off)
                         'use_tactile_gate': use_tactile_gate,
                         'tactile_gate_hidden': tactile_gate_hidden,
                         'tactile_gate_pool': tactile_gate_pool,
                         'tactile_gate_source': tactile_gate_source,
                         'tactile_gate_force_mode': tactile_gate_force_mode,
                         'tactile_gate_force_axis': tactile_gate_force_axis,

                         # Step-8.3: supervised gate loss (default off)
                         'tactile_gate_loss_weight': tactile_gate_loss_weight,

                         # ===== Step-9 =====
                         # Phase-A extras (data-only; still pass flags for model head config)
                         'step9_load_eef_ft_abs_obs': step9_load_eef_ft_abs_obs,
                         'ft_framestack': ft_framestack,
                         'step9_load_tactile_force6d_obs': step9_load_tactile_force6d_obs,
                         'step9_force_token_inject': step9_force_token_inject,
                         'step9_load_eef_ft_abs_future': step9_load_eef_ft_abs_future,
                         'step9_load_tactile_force6d_future': step9_load_tactile_force6d_future,
                         'step9_predict_horizon': step9_predict_horizon,
                         # Phase-B/9.3: aux loss knob (default-off)
                         'step9_future_force_loss_weight': step9_future_force_loss_weight,

                         # ===== Step11.1 (default off) =====
                         'step11_future_force_loss_weight': step11_future_force_loss_weight,
                         'step11_tactile_pool': step11_tactile_pool,

                         # ===== Step13.4 aux loss (default off) =====
                         'step13_aux_future_dw_loss_weight': step13_aux_future_dw_loss_weight,
                         }
    else:
        raise ValueError(f"Unsupported policy_class={policy_class!r}. This repo only supports ACT.")

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'qpos_dim': qpos_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        # 'real_robot': not is_sim,
        'action_mode': action_mode,
        'resume_path': resume_path,
        'use_tactile': use_tactile,
        'tactile_use_depth': tactile_use_depth,
        'tactile_use_img': tactile_use_img,
        'tactile_use_marker_offset': tactile_use_marker_offset,
        'tactile_use_force6d': tactile_use_force6d,
        'tactile_use_eef_ft': tactile_use_eef_ft,
        'tactile_use_joint_current': tactile_use_joint_current,
        'tactile_lowdim_mode': tactile_lowdim_mode,
        'tactile_hands': tactile_hands,
        'tactile_framestack': tactile_framestack,
        'tactile_vision_grid_size': tactile_vision_grid_size,
        'tactile_fusion': tactile_fusion,
        'tactile_delta': tactile_delta,
        'tactile_delta_clip': tactile_delta_clip,
        'tactile_delta_quantile_prints': tactile_delta_quantile_prints,
        'cudnn_benchmark': cudnn_benchmark,
        'dataloader_num_workers': dataloader_num_workers,
        'dataloader_pin_memory': dataloader_pin_memory,
        'dataloader_persistent_workers': dataloader_persistent_workers,
        'dataloader_prefetch_factor': dataloader_prefetch_factor,
        'vision_random_affine': args.get('vision_random_affine', False),
        'vision_photometric_noise': args.get('vision_photometric_noise', False),
        'use_tactile_residual_inject': use_tactile_residual_inject,
        'tactile_residual_scale': tactile_residual_scale,
        'vision_contact_noise_std': vision_contact_noise_std,
        'anti_freeze_weight': anti_freeze_weight,
        'use_tactile_gate': use_tactile_gate,
        'tactile_gate_source': tactile_gate_source,
        'tactile_gate_loss_weight': tactile_gate_loss_weight,
        'gate_diag_interval': gate_diag_interval,
        'gate_diag_max_batches': gate_diag_max_batches,
        'gate_diag_out_dir': gate_diag_out_dir,
        # Step-6.1: cross-attn export (only effective when tactile_fusion == 'cross_bi')
        'export_cross_attn': export_cross_attn,
        'export_cross_attn_interval': export_cross_attn_interval,
        'export_cross_attn_dir': export_cross_attn_dir,
        'export_cross_attn_layer_idx': export_cross_attn_layer_idx,

        # Step9 Phase-A targets (data-only)
        'step9_enable_targets': step9_enable_targets,
        'step9_load_eef_ft_abs_obs': step9_load_eef_ft_abs_obs,
        'ft_framestack': ft_framestack,
        'step9_load_tactile_force6d_obs': step9_load_tactile_force6d_obs,
        'step9_load_eef_ft_abs_future': step9_load_eef_ft_abs_future,
        'step9_load_tactile_force6d_future': step9_load_tactile_force6d_future,
        'step9_predict_horizon': step9_predict_horizon,
        'step9_future_force_loss_weight': step9_future_force_loss_weight,
        # Step11.1 aux loss (shares Step9 future targets)
        'step11_future_force_loss_weight': step11_future_force_loss_weight,
        'step11_tactile_pool': step11_tactile_pool,
        # Step9 Phase-B: force token injection location (Tavla-aligned)
        'step9_force_token_inject': step9_force_token_inject,
        # websocket eval mode (default off)
        'ws_mode': str(args.get('ws_mode', 'off')),
        'ws_host': str(args.get('ws_host', '0.0.0.0')),
        'ws_port': int(args.get('ws_port', 8765)),
    }

    if is_eval:
        ckpt_names = args.get('ckpt_names', ['policy_best.ckpt'])
        if isinstance(ckpt_names, str):
            ckpt_names = [ckpt_names]
        if not isinstance(ckpt_names, (list, tuple)) or len(ckpt_names) == 0:
            raise ValueError("--ckpt_names must provide at least one checkpoint filename")
        # ckpt_names = ['policy_epoch_3000_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # ########## Guard: refuse to overwrite existing training run ########## #
    if resume_path is None:
        _existing_ckpts = glob.glob(os.path.join(ckpt_dir, 'policy_epoch_*_seed_*.ckpt'))
        if _existing_ckpts or os.path.exists(os.path.join(ckpt_dir, 'policy_best.ckpt')):
            print(f"\n[ERROR] ckpt_dir 已存在训练产物，拒绝覆盖: {ckpt_dir}")
            print(f"  若要恢复训练，请使用 --resume_path 指定 checkpoint")
            print(f"  若要重新训练，请手动删除该目录或指定新的 --ckpt_dir")
            sys.exit(1)

    # ########## Step3：Load Data 只在train阶段 ########## #
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        vision_random_affine=args.get('vision_random_affine', False),
        vision_photometric_noise=args.get('vision_photometric_noise', False),
        action_mode=action_mode,
        use_tactile=use_tactile,
        tactile_use_depth=tactile_use_depth,
        tactile_use_img=tactile_use_img,
        tactile_use_marker_offset=tactile_use_marker_offset,
        tactile_use_force6d=tactile_use_force6d,
        tactile_use_eef_ft=tactile_use_eef_ft,
        tactile_use_joint_current=tactile_use_joint_current,
        tactile_hands=tactile_hands,
        tactile_framestack=tactile_framestack,
        tactile_gate_return_meta=tactile_gate_return_meta or (tactile_gate_loss_weight > 0),
        tactile_gate_key=tactile_gate_key,
        tactile_gate_load_force6d=tactile_gate_load_force6d,
        tactile_gate_load_marker_offset=tactile_gate_load_marker_offset,

        step9_enable_targets=step9_enable_targets,
        step9_load_eef_ft_abs_obs=step9_load_eef_ft_abs_obs,
        ft_framestack=ft_framestack,
        step9_load_tactile_force6d_obs=step9_load_tactile_force6d_obs,
        step9_load_eef_ft_abs_future=step9_load_eef_ft_abs_future,
        step9_predict_horizon=step9_predict_horizon,
        step9_load_tactile_force6d_future=step9_load_tactile_force6d_future,

        step13_enable_eef_wrench_hist=step13_enable_eef_wrench_hist,
        step13_force_source=step13_force_source,
        step13_residual_source=step13_residual_source,
        step13_residual_tactile_source=step13_residual_tactile_source,
        step13_residual_tactile_framestack=step13_residual_tactile_framestack,
        step13_enable_aux_targets=step13_aux_enabled,
        step13_aux_predict_horizon=step13_aux_predict_horizon,
        step13_aux_dw_mode=step13_aux_dw_mode,
        step13_aux_dw_space=step13_aux_dw_space,
        step13_aux_dw_norm_stats=step13_aux_dw_norm_stats,
    )

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # save config (training snapshot)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class != 'ACT':
        raise ValueError(f"Unsupported policy_class={policy_class!r}. This repo only supports ACT.")
    return ACTPolicy(policy_config)


def make_optimizer(policy_class, policy):
    if policy_class != 'ACT':
        raise ValueError(f"Unsupported policy_class={policy_class!r}. This repo only supports ACT.")
    return policy.configure_optimizers()

def get_image(obs, camera_names, device):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


class _TactileFrameStack:
    def __init__(
        self,
        *,
        tactile_hands=("left", "right"),
        tactile_modalities=("depth",),
        framestack: int = 1,
        default_hw=(128, 128),
    ) -> None:
        self.tactile_hands = tuple(tactile_hands)
        self.tactile_modalities = tuple(tactile_modalities)
        self.framestack = int(framestack)
        if self.framestack < 1:
            raise ValueError(f"framestack must be >= 1, got {self.framestack}")

        self._h = int(default_hw[0])
        self._w = int(default_hw[1])

        self._buf = {
            hand: {mod: deque(maxlen=self.framestack) for mod in self.tactile_modalities}
            for hand in self.tactile_hands
        }

    def _maybe_update_hw(self, pack) -> None:
        if not isinstance(pack, dict):
            return
        d = pack.get("depth", None)
        if isinstance(d, np.ndarray) and d.ndim == 2:
            self._h, self._w = int(d.shape[0]), int(d.shape[1])
            return
        im = pack.get("img", None)
        if isinstance(im, np.ndarray) and im.ndim == 3:
            self._h, self._w = int(im.shape[0]), int(im.shape[1])

    def _zero_frame(self, modality: str) -> np.ndarray:
        h, w = int(self._h), int(self._w)
        if modality == "depth":
            return np.zeros((h, w), dtype=np.float32)
        if modality == "img":
            return np.zeros((h, w, 3), dtype=np.uint8)
        if modality == "marker_offset":
            return np.zeros((9, 9, 2), dtype=np.float32)
        if modality == "force6d":
            return np.zeros((12,), dtype=np.float32)
        raise KeyError(f"unknown tactile modality: {modality!r}")

    @staticmethod
    def _sanitize_marker_offset(mo):
        if mo is None:
            return None
        mo = np.asarray(mo)
        if mo.ndim == 3 and mo.shape == (9, 9, 2):
            return mo.astype(np.float32, copy=False)
        if mo.ndim == 1 and mo.size == 9 * 9 * 2:
            return mo.astype(np.float32, copy=False).reshape(9, 9, 2)
        return None

    @staticmethod
    def _sanitize_force6d(f):
        if f is None:
            return None
        f = np.asarray(f, dtype=np.float32).reshape(-1)
        if f.shape[0] == 12:
            return f
        if f.shape[0] == 6:
            return np.concatenate([f, np.zeros_like(f)], axis=0)
        return None

    def update(self, raw_tactile) -> None:
        for hand in self.tactile_hands:
            pack = raw_tactile.get(hand, None) if isinstance(raw_tactile, dict) else None
            self._maybe_update_hw(pack if isinstance(pack, dict) else None)

            for mod in self.tactile_modalities:
                frame = None
                if isinstance(pack, dict):
                    frame = pack.get(mod, None)

                if mod == "depth":
                    if isinstance(frame, np.ndarray):
                        arr = np.asarray(frame)
                        if arr.ndim == 3 and arr.shape[-1] == 1:
                            arr = arr[..., 0]
                        if arr.ndim == 2:
                            frame = arr.astype(np.float32, copy=False)
                        else:
                            frame = None
                    else:
                        frame = None
                elif mod == "img":
                    if isinstance(frame, np.ndarray):
                        arr = np.asarray(frame)
                        if arr.ndim == 3 and arr.shape[-1] == 3:
                            frame = arr
                        else:
                            frame = None
                    else:
                        frame = None
                elif mod == "marker_offset":
                    frame = self._sanitize_marker_offset(frame) if isinstance(frame, np.ndarray) else None
                elif mod == "force6d":
                    frame = self._sanitize_force6d(frame) if isinstance(frame, np.ndarray) else None
                else:
                    raise KeyError(f"unknown tactile modality: {mod!r}")

                if frame is None:
                    frame = self._zero_frame(mod)
                self._buf[hand][mod].append(frame)

    @staticmethod
    def _pad_to_framestack(buf: deque, t: int, default: np.ndarray):
        if len(buf) == 0:
            return [default for _ in range(t)]
        frames = list(buf)
        while len(frames) < t:
            frames.insert(0, frames[-1])
        if len(frames) > t:
            frames = frames[-t:]
        return frames

    def get_stacked(self) -> dict:
        out = {}
        for hand in self.tactile_hands:
            hand_out = {}
            for mod in self.tactile_modalities:
                default = self._zero_frame(mod)
                frames = self._pad_to_framestack(self._buf[hand][mod], self.framestack, default)
                hand_out[mod] = np.stack(frames, axis=0)
            out[hand] = hand_out
        return out


def get_tactile(
    obs,
    *,
    tactile_hands=("left", "right"),
    tactile_use_depth=True,
    tactile_use_img=False,
    tactile_use_marker_offset=False,
    tactile_use_force6d=False,
    step13_residual_tactile_source=None,
    step13_residual_tactile_obs=None,
    device=None,
    default_hw=(128, 128),
):
    """Convert env tactile payload (numpy) into training-compatible torch dict.

    Returns: tactile dict shaped like dataset output:
      {hand: {"depth": [B,1,H,W], "img": [B,3,H,W], ...}}
    """

    raw = obs.get("tactile", None)

    def _infer_hw():
        if isinstance(raw, dict):
            for _, pack in raw.items():
                if not isinstance(pack, dict):
                    continue
                d = pack.get("depth", None)
                if isinstance(d, np.ndarray) and d.ndim == 2:
                    return (int(d.shape[0]), int(d.shape[1]))
                im = pack.get("img", None)
                if isinstance(im, np.ndarray) and im.ndim == 3:
                    return (int(im.shape[0]), int(im.shape[1]))
        return (int(default_hw[0]), int(default_hw[1]))

    h, w = _infer_hw()
    tactile = {}
    for hand in tuple(tactile_hands):
        pack = raw.get(hand, None) if isinstance(raw, dict) else None
        hand_out = {}

        if tactile_use_depth:
            depth = pack.get("depth", None) if isinstance(pack, dict) else None
            if not isinstance(depth, np.ndarray):
                depth = np.zeros((h, w), dtype=np.float32)
            depth_t = torch.from_numpy(depth).float()
            if depth_t.ndim == 2:
                depth_t = depth_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            elif depth_t.ndim == 3:
                depth_t = depth_t.unsqueeze(0)  # [1,1,H,W] or [1,T,H,W]
            elif depth_t.ndim == 4:
                pass
            else:
                raise ValueError(f"unexpected tactile depth shape: {tuple(depth_t.shape)}")
            hand_out["depth"] = depth_t

        if tactile_use_img:
            img = pack.get("img", None) if isinstance(pack, dict) else None
            if not isinstance(img, np.ndarray):
                img = np.zeros((h, w, 3), dtype=np.uint8)
            img_t = torch.from_numpy(img)
            if img_t.ndim == 3:
                img_t = torch.einsum("h w c -> c h w", img_t)
                img_t = img_t.unsqueeze(0)  # [1,3,H,W]
            elif img_t.ndim == 4:
                # [T,H,W,C] -> [1,T,3,H,W]
                img_t = torch.einsum("t h w c -> t c h w", img_t)
                img_t = img_t.unsqueeze(0)
            is_uint8 = img_t.dtype == torch.uint8
            img_t = img_t.float()
            if is_uint8:
                img_t = img_t / 255.0
            hand_out["img"] = img_t

        if tactile_use_marker_offset:
            mo = pack.get("marker_offset", None) if isinstance(pack, dict) else None
            if not isinstance(mo, np.ndarray):
                mo = np.zeros((9, 9, 2), dtype=np.float32)
            mo = np.asarray(mo, dtype=np.float32)
            # allow: [9,9,2] or [T,9,9,2] (framestack)
            if mo.ndim == 1 and mo.size == 9 * 9 * 2:
                mo = mo.reshape(9, 9, 2)
            if mo.ndim not in (3, 4):
                raise ValueError(f"unexpected marker_offset shape: {tuple(mo.shape)}")
            if mo.ndim == 3 and mo.shape != (9, 9, 2):
                raise ValueError(f"unexpected marker_offset shape: {tuple(mo.shape)}")
            if mo.ndim == 4 and mo.shape[-3:] != (9, 9, 2):
                raise ValueError(f"unexpected marker_offset shape: {tuple(mo.shape)}")

            mo_t = torch.from_numpy(mo).float()
            if mo_t.ndim == 3:
                mo_t = mo_t.unsqueeze(0)  # [1,9,9,2]
            elif mo_t.ndim == 4:
                mo_t = mo_t.unsqueeze(0)  # [1,T,9,9,2]
            hand_out["marker_offset"] = mo_t

        if tactile_use_force6d:
            f = pack.get("force6d", None) if isinstance(pack, dict) else None
            if not isinstance(f, np.ndarray):
                f12 = np.zeros((12,), dtype=np.float32)
                hand_out["force6d"] = torch.from_numpy(f12).float().unsqueeze(0)  # [1,12]
            else:
                f = np.asarray(f, dtype=np.float32)
                if f.ndim == 1:
                    ff = f.reshape(-1)
                    if ff.shape[0] == 12:
                        f12 = ff
                    elif ff.shape[0] == 6:
                        f12 = np.concatenate([ff, np.zeros_like(ff)], axis=0)
                    else:
                        raise ValueError(f"unexpected force6d shape: {tuple(ff.shape)}")
                    hand_out["force6d"] = torch.from_numpy(f12).float().unsqueeze(0)  # [1,12]
                elif f.ndim == 2:
                    t, d = f.shape
                    if d == 12:
                        f12 = f
                    elif d == 6:
                        zeros = np.zeros((t, 6), dtype=np.float32)
                        f12 = np.concatenate([f, zeros], axis=1)
                    else:
                        raise ValueError(f"unexpected force6d shape: {tuple(f.shape)}")
                    hand_out["force6d"] = torch.from_numpy(f12).float().unsqueeze(0)  # [1,T,12]
                elif f.ndim == 3:
                    # assume [B,T,12] or [B,T,6]
                    b, t, d = f.shape
                    if d == 12:
                        f12 = f
                    elif d == 6:
                        zeros = np.zeros((b, t, 6), dtype=np.float32)
                        f12 = np.concatenate([f, zeros], axis=2)
                    else:
                        raise ValueError(f"unexpected force6d shape: {tuple(f.shape)}")
                    hand_out["force6d"] = torch.from_numpy(f12).float()
                else:
                    raise ValueError(f"unexpected force6d shape: {tuple(f.shape)}")

        tactile[hand] = hand_out

    if step13_residual_tactile_source is not None:
        residual_raw_obs = step13_residual_tactile_obs if step13_residual_tactile_obs is not None else obs
        residual_raw = residual_raw_obs.get("tactile", None) if isinstance(residual_raw_obs, dict) else None
        residual_pack = {}
        for hand in tuple(tactile_hands):
            pack = residual_raw.get(hand, None) if isinstance(residual_raw, dict) else None
            hand_out = {}

            if step13_residual_tactile_source == "depth":
                depth = pack.get("depth", None) if isinstance(pack, dict) else None
                if not isinstance(depth, np.ndarray):
                    depth = np.zeros((h, w), dtype=np.float32)
                depth_t = torch.from_numpy(depth).float()
                if depth_t.ndim == 2:
                    depth_t = depth_t.unsqueeze(0).unsqueeze(0)
                elif depth_t.ndim == 3:
                    depth_t = depth_t.unsqueeze(0)
                elif depth_t.ndim != 4:
                    raise ValueError(
                        f"unexpected residual tactile depth shape: {tuple(depth_t.shape)}"
                    )
                hand_out["depth"] = depth_t

            elif step13_residual_tactile_source == "img":
                img = pack.get("img", None) if isinstance(pack, dict) else None
                if not isinstance(img, np.ndarray):
                    img = np.zeros((h, w, 3), dtype=np.uint8)
                img_t = torch.from_numpy(img)
                if img_t.ndim == 3:
                    img_t = torch.einsum("h w c -> c h w", img_t)
                    img_t = img_t.unsqueeze(0)
                elif img_t.ndim == 4:
                    img_t = torch.einsum("t h w c -> t c h w", img_t)
                    img_t = img_t.unsqueeze(0)
                else:
                    raise ValueError(
                        f"unexpected residual tactile img shape: {tuple(img_t.shape)}"
                    )
                is_uint8 = img_t.dtype == torch.uint8
                img_t = img_t.float()
                if is_uint8:
                    img_t = img_t / 255.0
                hand_out["img"] = img_t

            elif step13_residual_tactile_source == "marker_offset":
                mo = pack.get("marker_offset", None) if isinstance(pack, dict) else None
                if not isinstance(mo, np.ndarray):
                    mo = np.zeros((9, 9, 2), dtype=np.float32)
                mo = np.asarray(mo, dtype=np.float32)
                if mo.ndim == 1 and mo.size == 9 * 9 * 2:
                    mo = mo.reshape(9, 9, 2)
                if mo.ndim not in (3, 4):
                    raise ValueError(
                        f"unexpected residual marker_offset shape: {tuple(mo.shape)}"
                    )
                mo_t = torch.from_numpy(mo).float()
                if mo_t.ndim in (3, 4):
                    mo_t = mo_t.unsqueeze(0)
                hand_out["marker_offset"] = mo_t

            else:
                raise ValueError(
                    "step13_residual_tactile_source must be one of "
                    "{'depth','img','marker_offset'}, "
                    f"got {step13_residual_tactile_source!r}"
                )

            residual_pack[hand] = hand_out

        tactile["step13_residual"] = residual_pack

    if device is not None:
        tactile = _tree_to(tactile, device)
    return tactile


def _self_test_tactile_framestack():
    # Minimal contract check against tactile encoders (shape + framestack).
    from detr.models.tactile_encoder import (
        TactileDepthMMTokenEncoder,
        TactileForce6DTokenEncoder,
        TactileImgVisionTokenEncoder,
        TactileMarkerOffsetTokenEncoder,
    )

    device = torch.device('cpu')
    t = 5
    h, w = 32, 32

    stacker = _TactileFrameStack(
        tactile_hands=("left", "right"),
        tactile_modalities=("depth", "img", "marker_offset", "force6d"),
        framestack=t,
        default_hw=(h, w),
    )

    for i in range(t):
        raw = {
            "left": {
                "depth": (np.ones((h, w), dtype=np.float32) * i),
                "img": (np.ones((h, w, 3), dtype=np.uint8) * (i * 10)),
                "marker_offset": (np.ones((9, 9, 2), dtype=np.float32) * i),
                "force6d": (np.arange(6, dtype=np.float32) + i),
            },
            "right": {
                "depth": (np.ones((h, w), dtype=np.float32) * i),
                "img": (np.ones((h, w, 3), dtype=np.uint8) * (i * 10)),
                "marker_offset": (np.ones((9, 9, 2), dtype=np.float32) * i),
                "force6d": (np.arange(12, dtype=np.float32) + i),
            },
        }
        stacker.update(raw)

    obs = {"tactile": stacker.get_stacked()}
    tactile = get_tactile(
        obs,
        tactile_hands=("left", "right"),
        tactile_use_depth=True,
        tactile_use_img=True,
        tactile_use_marker_offset=True,
        tactile_use_force6d=True,
        device=device,
        default_hw=(h, w),
    )

    assert tactile["left"]["depth"].shape == (1, t, h, w)
    assert tactile["left"]["img"].shape == (1, t, 3, h, w)
    assert tactile["left"]["marker_offset"].shape == (1, t, 9, 9, 2)
    assert tactile["left"]["force6d"].shape == (1, t, 12)

    # Run each encoder forward to ensure framestack checks pass.
    batch_size = 1
    enc_d = TactileDepthMMTokenEncoder(hidden_dim=32, framestack=t).to(device)
    enc_i = TactileImgVisionTokenEncoder(hidden_dim=32, framestack=t, grid_size=-1).to(device)
    enc_m = TactileMarkerOffsetTokenEncoder(hidden_dim=32, framestack=t).to(device)
    enc_f = TactileForce6DTokenEncoder(hidden_dim=32, framestack=t).to(device)

    _ = enc_d(tactile, batch_size=batch_size)
    _ = enc_i(tactile, batch_size=batch_size)
    _ = enc_m(tactile, batch_size=batch_size)
    _ = enc_f(tactile, batch_size=batch_size)

    print("[imitate_episodes] tactile_self_test OK")


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    # state_dim: 动作维度；qpos_dim: 观测关节维度
    state_dim = config['state_dim']
    qpos_dim = config.get('qpos_dim', state_dim)
    # real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    ws_mode = str(config.get('ws_mode', 'off'))
    ws_host = str(config.get('ws_host', '0.0.0.0'))
    ws_port = int(config.get('ws_port', 8765))

    if ws_mode not in ('off', 'server'):
        raise ValueError(f"Unsupported ws_mode={ws_mode!r}, expected 'off' or 'server'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    loading_status = policy.load_state_dict(checkpoint)
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment (local mode) or websocket server bridge (remote mode)
    env = None
    ws_server = None
    env_cfg = None

    # 根据训练时的动作模式配置动作维度
    action_mode = config.get('action_mode', 'joint')
    expected_action_dim = 7 if action_mode == 'joint' else 6

    # tactile: 部署端采集配置已在 RealmanEnv.Config 中写死（相机/左右手/4模态）；
    # 这里仅保留“模型侧”开关，用于决定喂给 policy 哪些模态。
    use_tactile = bool(config.get('use_tactile', False))
    tactile_use_depth = bool(config.get('tactile_use_depth', True))
    tactile_use_img = bool(config.get('tactile_use_img', False))
    tactile_use_marker_offset = bool(config.get('tactile_use_marker_offset', False))
    tactile_use_force6d = bool(config.get('tactile_use_force6d', False))
    tactile_hands = tuple(config.get('tactile_hands', ['left', 'right']))
    tactile_framestack = int(config.get('tactile_framestack', 1))
    step13_enabled = bool(config.get('step13_enabled', False))
    step13_config = config.get('step13_config', {})
    step13_residual_runtime = _resolve_step13_residual_tactile_runtime(step13_config)
    step13_residual_enabled = bool(step13_residual_runtime['enabled'])
    step13_force_source = _normalize_step13_force_source(
        step13_config.get('force_source', 'eef_ft') if isinstance(step13_config, dict) else 'eef_ft'
    )
    ft_framestack = int(policy_config.get('ft_framestack', 1) or 1)

    tactile_modalities = []
    if tactile_use_depth:
        tactile_modalities.append('depth')
    if tactile_use_img:
        tactile_modalities.append('img')
    if tactile_use_marker_offset:
        tactile_modalities.append('marker_offset')
    if tactile_use_force6d:
        tactile_modalities.append('force6d')

    tactile_stacker = None
    if use_tactile and tactile_framestack > 1 and len(tactile_modalities) > 0:
        tactile_stacker = _TactileFrameStack(
            tactile_hands=tactile_hands,
            tactile_modalities=tuple(tactile_modalities),
            framestack=tactile_framestack,
            default_hw=(128, 128),
        )
    residual_tactile_stacker = None
    if step13_residual_enabled and int(step13_residual_runtime['tactile_framestack']) > 1:
        residual_tactile_stacker = _TactileFrameStack(
            tactile_hands=tactile_hands,
            tactile_modalities=(str(step13_residual_runtime['tactile_source']),),
            framestack=int(step13_residual_runtime['tactile_framestack']),
            default_hw=(128, 128),
        )
    step13_hist = deque(maxlen=ft_framestack) if step13_enabled else None

    if ws_mode == 'off':
        env_cfg = RealmanEnvConfig()
        env_cfg.ACTION_MODE = 'eef_rel' if action_mode == 'eef_rel' else 'joint'

        # 可选：实时可视化六维力/矩（ft）
        if config.get('visualize_ft', False):
            env_cfg.VISUALIZE_FT = True
            env_cfg.VISUALIZE_FT_HZ = float(config.get('visualize_ft_hz', 20))
            env_cfg.VISUALIZE_FT_WINDOW = int(config.get('visualize_ft_window', max_timesteps))

        # 可选：保存六维力/矩（ft），长度严格等于 episode 步数
        if config.get('save_ft', False):
            env_cfg.SAVE_FT = True

        env = RealmanEnv(env_cfg)
    else:
        from tools.websocket.miact_ws_server import ClientDisconnected
        from tools.websocket.miact_ws_server import MIACTWebsocketActionServer

        ws_server = MIACTWebsocketActionServer(host=ws_host, port=ws_port)
        ws_server.start()
        print(f"[imitate_episodes] WebSocket server mode enabled: ws://{ws_host}:{ws_port}")
    env_max_reward = 0

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 1
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        print(f"Rollout begins!")
        try:
            if ws_mode == 'off':
                obs = env.reset()##
            else:
                print("[imitate_episodes] Waiting first client obs...")
                obs = ws_server.recv_obs()
        except ClientDisconnected:
            print("[imitate_episodes] client disconnected before first obs; abort this rollout")
            break
        if use_tactile and "tactile" not in obs:
            print(
                "[imitate_episodes] ⚠️ use_tactile=True 但 obs 未返回触觉数据；将以 0 张量占位。"
            )

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps+num_queries, state_dim],
                device=device,
            )

        qpos_history = torch.zeros((1, max_timesteps, qpos_dim), device=device)
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                try:
                    print(f'Timestep {t}/{max_timesteps}', end='\r')
                    image_list.append(obs['images'])

                    if tactile_stacker is not None:
                        tactile_stacker.update(obs.get('tactile', None))
                    if residual_tactile_stacker is not None:
                        residual_tactile_stacker.update(obs.get('tactile', None))
                    if step13_hist is not None:
                        _update_step13_hist_runtime(
                            step13_hist,
                            obs,
                            force_source=step13_force_source,
                        )

                    qpos_numpy = np.array(obs['proprio'])
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                    # qpos 维度通常为 (1, qpos_dim)
                    qpos_history[:, t, :qpos.shape[-1]] = qpos
                    curr_image = get_image(obs, camera_names, device)

                    ### query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            tactile = None
                            if use_tactile or step13_enabled or step13_residual_enabled:
                                obs_for_tactile = obs
                                obs_for_step13_residual = obs
                                if tactile_stacker is not None:
                                    obs_for_tactile = dict(obs)
                                    obs_for_tactile['tactile'] = tactile_stacker.get_stacked()
                                if residual_tactile_stacker is not None:
                                    obs_for_step13_residual = dict(obs)
                                    obs_for_step13_residual['tactile'] = residual_tactile_stacker.get_stacked()
                                tactile = get_tactile(
                                    obs_for_tactile,
                                    tactile_hands=tactile_hands,
                                    tactile_use_depth=tactile_use_depth,
                                    tactile_use_img=tactile_use_img,
                                    tactile_use_marker_offset=tactile_use_marker_offset,
                                    tactile_use_force6d=tactile_use_force6d,
                                    step13_residual_tactile_source=step13_residual_runtime['tactile_source'] if step13_residual_enabled else None,
                                    step13_residual_tactile_obs=obs_for_step13_residual,
                                    device=device,
                                )
                                if step13_enabled:
                                    step13_force_pack = _step13_hist_to_force_pack_runtime(
                                        step13_hist,
                                        hist_len=ft_framestack,
                                        stats=stats,
                                        device=device,
                                        force_source=step13_force_source,
                                    )
                                    eef = tactile.setdefault('eef', {})
                                    if step13_force_source == 'eef_ft':
                                        ft_pack = eef.setdefault('ft', {})
                                        ft_pack['abs'] = step13_force_pack['abs']
                                    step13_pack = eef.setdefault('step13_force', {})
                                    step13_pack.update(step13_force_pack)
                            all_actions = policy(qpos, curr_image, tactile=tactile)
                        if temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
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
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action

                    action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
                    if action_vec.shape[0] != expected_action_dim:
                        raise ValueError(
                            f"Action dim mismatch: expected {expected_action_dim} for action_mode={action_mode}, got {action_vec.shape}"
                        )

                    ### step the environment (local) or remote client (websocket)
                    if ws_mode == 'off':
                        obs, rew = env.step(action_vec)
                    else:
                        ws_server.send_action(action_vec)
                        rew = 0.0
                        if t + 1 < max_timesteps:
                            obs = ws_server.recv_obs()
                except ClientDisconnected:
                    print("[imitate_episodes] client disconnected mid-rollout; ending rollout early")
                    break

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(rew)

            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

        # 可选：保存 ft (T,6) 到 ckpt_dir（仅本地 env 模式）
        if config.get('save_ft', False) and ws_mode == 'off':
            try:
                ft_arr = env.get_episode_ft() if hasattr(env, 'get_episode_ft') else None
                if ft_arr is not None:
                    out_path = config.get('save_ft_path', None)
                    ts = time.strftime('%Y%m%d_%H%M%S')

                    # out_path:
                    # - None: 默认写到 ckpt_dir
                    # - 目录: 写到该目录
                    # - 文件名: 按用户指定
                    if out_path is None:
                        out_path = os.path.join(ckpt_dir, f'ft_rollout{rollout_id}_{ts}.npy')
                    else:
                        out_path = str(out_path)
                        if (os.path.exists(out_path) and os.path.isdir(out_path)) or out_path.endswith('/') or out_path.endswith('\\'):
                            out_dir = out_path
                            out_path = os.path.join(out_dir, f'ft_rollout{rollout_id}_{ts}.npy')

                    ft_to_save = ft_arr if ft_arr.dtype == np.float32 else ft_arr.astype(np.float32)
                    np.save(out_path, ft_to_save)
                    print(f"Saved ft to: {out_path}, shape={ft_arr.shape}")
            except Exception as e:
                print("[imitate_episodes] ⚠️ 保存 ft 失败:", e)

    if ws_server is not None:
        ws_server.close()

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy, device=None):
    if device is None:
        try:
            device = next(policy.parameters()).device  # 获取模型参数所在的设备
        except StopIteration:  # 模型没有参数
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _looks_like_tactile_pack(x) -> bool:
        # Tactile pack must be a dict and should not look like legacy Step9 payload.
        if not isinstance(x, dict):
            return False
        if any(isinstance(k, str) and k.startswith("step9_") for k in x.keys()):
            return False
        for hand in ("left", "right"):
            if hand in x and isinstance(x[hand], dict):
                return True
        return False

    tactile = None
    if len(data) == 4:
        image_data, qpos_data, action_data, is_pad = data
    elif len(data) == 5:
        image_data, qpos_data, action_data, is_pad, x5 = data
        if not _looks_like_tactile_pack(x5):
            raise ValueError(
                "Expected tactile pack as 5th batch item; "
                f"got keys={list(x5.keys()) if isinstance(x5, dict) else type(x5)}"
            )
        tactile = x5
    else:
        raise ValueError(f"Unexpected batch size {len(data)}")

    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)

    if tactile is not None:
        tactile = _tree_to(tactile, device)

    return policy(
        qpos_data,
        image_data,
        action_data,
        is_pad,
        tactile=tactile,
    )


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    resume_path = config.get('resume_path', None)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    start_epoch = 0

    def _fmt_epoch_loss(prefix: str, summary: dict) -> str:
        def _get(name: str) -> float:
            v = summary.get(name, None)
            try:
                return float(v.item())
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return float('nan')

        l1 = _get('l1')
        kl = _get('kl')
        anti_freeze = _get('anti_freeze')
        gate_bce = _get('gate_bce')
        step9_future_force = _get('step9_future_force')
        total = _get('loss')

        # NOTE: weights live in policy_config (and that's what the policy uses).
        # config top-level does not carry these fields, so reading from there
        # would print misleading zeros.
        pcfg = policy_config if isinstance(policy_config, dict) else config.get('policy_config', {})
        kl_w = float(pcfg.get('kl_weight', 0.0))
        af_w = float(pcfg.get('anti_freeze_weight', 0.0))
        gate_w = float(pcfg.get('tactile_gate_loss_weight', 0.0))
        fut_w = float(pcfg.get('step9_future_force_loss_weight', 0.0))

        return (
            f"{prefix} loss={total:.6f} "
            f"| l1={l1:.6f} "
            f"| kl={kl:.6f} (x{kl_w:g} => {kl*kl_w:.6f}) "
            f"| anti_freeze={anti_freeze:.6f} (x{af_w:g} => {anti_freeze*af_w:.6f}) "
            f"| gate_bce={gate_bce:.6f} (x{gate_w:g} => {gate_bce*gate_w:.6f}) "
            f"| step9_future_force={step9_future_force:.6f} (x{fut_w:g} => {step9_future_force*fut_w:.6f})"
        )

    # === resume ===
    if resume_path is not None and os.path.exists(resume_path):
        # 不能是 policy_best、policy_last，因为这些只保存了 model.state_dict()
        print(f"[Resume] Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        min_val_loss = float(checkpoint.get('min_val_loss', np.inf))
        best_epoch = int(checkpoint.get('best_epoch', -1))
        best_state_dict = torch.load(os.path.join(ckpt_dir, 'policy_best.ckpt'), map_location=device)
        best_ckpt_info = (best_epoch, min_val_loss, best_state_dict)
        print(f"[Resume] start_epoch={start_epoch}, min_val_loss={min_val_loss}")

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')

        from tools.gate.gate_diag_runtime import GateDiagConfig, GateDiagRuntime

        gate_diag = GateDiagRuntime(
            ckpt_dir=ckpt_dir,
            cfg=GateDiagConfig(
                interval=int(config.get('gate_diag_interval', 0) or 0),
                max_batches=int(config.get('gate_diag_max_batches', 5) or 0),
                out_dir=config.get('gate_diag_out_dir', None),
            ),
        )
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = maybe_export_layer0_cross_attn(
                    policy=policy,
                    data=data,
                    forward_fn=forward_pass,
                    device=device,
                    config=config,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                torch.save(best_ckpt_info[2], os.path.join(ckpt_dir, 'policy_best.ckpt'))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        print(_fmt_epoch_loss('Val', epoch_summary))

        # training
        policy.train()
        optimizer.zero_grad()
        epoch_train_dicts = []
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device=device)

            gate_diag.collect_from_batch(epoch=epoch, batch_idx=batch_idx, data=data, policy=policy)

            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # train_history.append(detach_dict(forward_dict))
            detached = detach_dict(forward_dict)
            epoch_train_dicts.append(detached)
            train_history.append(detached)

        gate_diag.flush_epoch(epoch=epoch)
        # epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_summary = compute_dict_mean(epoch_train_dicts)
        epoch_train_loss = epoch_summary['loss'].item()
        print(f'Train loss: {epoch_train_loss:.5f}')
        print(_fmt_epoch_loss('Train', epoch_summary))

        if epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save({
                'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'min_val_loss': float(min_val_loss),
                'best_epoch': int(best_ckpt_info[0]),
            }, ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        # plt.tight_layout()  # 导致图例显示不全
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    # print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument(
        '--ckpt_names',
        nargs='+',
        default=['policy_best.ckpt'],
        help='eval模式下要评估的ckpt文件名列表（默认: policy_best.ckpt）'
    )
    parser.add_argument(
        '--exp_config',
        type=str,
        default=None,
        help='实验配置 JSON（支持 step9/step13 两个独立 section；未知 key 直接报错）',
    )

    # ===== Dataset/task config (required; constants.py deprecated) =====
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='离线数据目录（包含 episode_*.hdf5）')
    parser.add_argument('--num_episodes', type=int, default=-1,
                        help='用于训练/验证的 episode 数；默认 -1 表示自动读取目录内最大 episode 编号+1')
    parser.add_argument('--episode_len', type=int, required=True,
                        help='每条轨迹最大长度（用于 rollout/eval 的 max steps）')
    parser.add_argument('--camera_names', nargs='+', type=str, required=True,
                        help='相机名称列表（对应 HDF5: /observations/images/<cam>），例如：global wrist')


    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument(
        '--pre_norm',
        action='store_true',
        help='显式启用 pre-norm Transformer（默认 post-norm；与 DETR/ACT legacy 行为一致）',
    )
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--action_mode', action='store', type=str, choices=['joint', 'eef_rel'], default='joint',
                        help='训练和部署时使用的动作表示：joint(关节绝对角) 或 eef_rel(末端增量)')
    parser.add_argument('--resume_path', type=str, default=None, help='path to resume checkpoint')

    # ===== 训练速度相关（默认值保持历史行为） =====
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=False,
                        help='开启 cuDNN benchmark 自动选最快卷积算法（默认关闭；输入尺寸稳定时可能提速）')

    # ===== DataLoader 速度相关超参（默认值保持历史行为） =====
    parser.add_argument('--dataloader_num_workers', type=int, default=1,
                        help='DataLoader num_workers（默认1；设为0可禁用多进程加载）')
    parser.add_argument('--dataloader_pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help='DataLoader pin_memory（默认开启；CPU->GPU 拷贝可能更快）')
    parser.add_argument('--dataloader_persistent_workers', action=argparse.BooleanOptionalAction, default=False,
                        help='DataLoader persistent_workers（默认关闭；num_workers>0 时可减少 epoch 间 worker 重启开销）')
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=1,
                        help='DataLoader prefetch_factor（默认1；仅在 num_workers>0 时生效）')

    # ===== Vision augmentation（默认关闭；仅训练集生效） =====
    parser.add_argument(
        '--vision_random_affine',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='训练集：视觉图像 RandomAffine 增强（默认关闭）',
    )
    parser.add_argument(
        '--vision_photometric_noise',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='训练集：视觉图像随机光照/对比度/gamma/噪声增强（默认关闭）',
    )

    # ===== tactile（Step-3 起：数据读取 + 模型侧可选接入，默认关闭） =====
    parser.add_argument('--use_tactile', action='store_true', help='Dataset 额外返回 tactile（默认关闭）')
    parser.add_argument('--tactile_use_depth', action=argparse.BooleanOptionalAction, default=True,
                        help='tactile 返回/使用 depth（Step-3 默认 True；可用 --no-tactile_use_depth 关闭）')
    parser.add_argument('--tactile_use_img', action='store_true', help='tactile 额外返回 img（默认关闭）')
    parser.add_argument('--tactile_use_marker_offset', action='store_true', help='tactile 额外返回 marker_offset（默认关闭）')
    parser.add_argument('--tactile_use_force6d', action='store_true', help='tactile 额外返回 force6d（默认关闭）')
    parser.add_argument('--tactile_use_eef_ft', action='store_true', help='tactile 额外返回 /ft 并作为全局 token（默认关闭）')
    parser.add_argument('--tactile_use_joint_current', action='store_true', help='tactile 额外返回 /joint_current 并作为全局 token（默认关闭）')
    parser.add_argument(
        '--tactile_lowdim_mode',
        type=str,
        default='absrel',
        choices=['absrel', 'absrel0', 'abs'],
        help=(
            '低维向量类触觉 token 特征模式：'
            'absrel=abs+相邻差分Δ（默认），absrel0=abs+对首帧差分，abs=只用abs(用0补齐rel保持维度不变)'
        ),
    )
    parser.add_argument('--tactile_depth_norm', type=str, default='amax', choices=['amax', 'none'],
                        help='Model-side depth 预处理：depth/amax 或不做（默认 amax）')
    parser.add_argument('--tactile_hands', nargs='+', default=['left', 'right'], choices=['left', 'right'],
                        help='tactile hands（左右指）：允许 left / right / left right（默认 left right）')

    parser.add_argument('--tactile_self_test', action='store_true',
                        help='离线自检：验证 tactile framestack 的 shape/encoder 契约（不跑 env）')

    # ===== Step-4.2 framestack + Step-4.1 tokenizers =====
    parser.add_argument('--tactile_framestack', type=int, default=1,
                        help='tactile framestack 历史长度H（默认1；参考 vitac 可设18）')
    parser.add_argument('--tactile_vision_grid_size', type=int, default=0,
                        help='tactile img tokens 网格大小：0=保留backbone原生特征图；-1=每手1个token(GAP)；N>0=每手N×N tokens')

    # ===== Step-7.1/7.2：Δtactile + clamp（默认关闭；一键回退） =====
    parser.add_argument('--tactile_delta', type=str, default='none', choices=['none', 'adjacent'],
                        help='将 tactile depth/img 转为相邻帧差分（framestack>1 时生效；marker_offset 不受影响）')
    parser.add_argument('--tactile_delta_clip', type=float, default=0.0,
                        help='对差分后的 depth/img 做 clamp 到 [-c,c]（0 表示关闭）')

    parser.add_argument('--tactile_delta_quantile_prints', type=int, default=5,
                        help='打印 |Δtactile| 的 p95/p99 的次数（仅在 tactile_delta=adjacent 且 framestack>1 时生效；0 表示不打印）')

    # ===== Step-7.3：anti-freeze（默认关闭；一键回退） =====
    parser.add_argument('--anti_freeze_weight', type=float, default=0.0,
                        help='增加时间差分 L1 正则：L += w * L1(Δa_hat, Δa_gt)（0 表示关闭）')

    # ===== Step-8：contact gate + supervised loss（默认关闭；一键回退） =====
    parser.add_argument('--use_tactile_gate', action='store_true',
                        help='启用 tactile contact gate（默认关闭）')
    parser.add_argument('--tactile_gate_hidden', type=int, default=32,
                        help='gate MLP hidden dim（默认32）')
    parser.add_argument('--tactile_gate_pool', type=str, default='last', choices=['last'],
                        help='gate pooling over time（目前仅支持 last）')
    parser.add_argument('--tactile_gate_source', type=str, default='force6d', choices=['force6d', 'marker_offset'],
                        help='gate 输入源（默认 force6d）')
    parser.add_argument('--tactile_gate_force_mode', type=str, default='axis', choices=['axis', 'norm'],
                        help='force6d gate 标量化方式（axis/norm；仅在 source=force6d 时用）')
    parser.add_argument('--tactile_gate_force_axis', type=int, default=2,
                        help='force6d gate 取哪一维（axis 模式；默认2）')

    parser.add_argument('--tactile_gate_loss_weight', type=float, default=0.0,
                        help='gate 监督 BCE loss 权重（0=关闭；Step8.3）')
    parser.add_argument('--tactile_gate_return_meta', action='store_true',
                        help='dataset 返回 tactile["_gate"]={t_onset,t_ref}（默认关闭；Step8.1/8.3）')
    parser.add_argument('--tactile_gate_key', type=str, default='/t_onset/stable_global',
                        help='HDF5 onset 标签 key（默认 /t_onset/stable_global）')
    parser.add_argument('--tactile_gate_load_force6d', action='store_true',
                        help='仅为 gate 输入加载 force6d（不启用 token encoder；默认关闭）')
    parser.add_argument('--tactile_gate_load_marker_offset', action='store_true',
                        help='仅为 gate 输入加载 marker_offset（不启用 token encoder；默认关闭）')

    # ===== Step-8 diag：训练期 gate 可观测性（默认关闭） =====
    parser.add_argument('--gate_diag_interval', type=int, default=0,
                        help='每隔多少个 epoch 导出一次 gate 诊断图（0=关闭）')
    parser.add_argument('--gate_diag_max_batches', type=int, default=5,
                        help='每次导出最多采样多少个 train batch（默认5）')
    parser.add_argument('--gate_diag_out_dir', type=str, default=None,
                        help='gate 诊断输出目录（默认 ckpt_dir/gate_diag）')

    # ===== Step-9 (Phase A)：数据侧额外返回 /ft abs 与未来监督目标（默认关闭；一键回退） =====
    parser.add_argument('--step9_load_eef_ft_abs_obs', action='store_true',
                        help='Step9：返回 eef /ft abs（归一化 raw6d）历史观测（默认关闭）')
    parser.add_argument('--ft_framestack', type=int, default=1,
                        help='共享：/ft abs 历史长度H（默认1）。Step9/Step13/以及 tactile_use_eef_ft token 均使用该长度')
    parser.add_argument('--step9_load_tactile_force6d_obs', action='store_true',
                        help='Step9：返回 tactile force6d（raw 6D/hand）历史观测（默认关闭）')
    parser.add_argument('--step9_predict_horizon', type=int, default=0,
                        help='Step9：未来监督目标长度H_pred（0=默认对齐 chunk_size）')
    parser.add_argument('--step9_load_eef_ft_abs_future', action='store_true',
                        help='Step9：返回 eef /ft abs 未来监督目标（默认关闭）')
    parser.add_argument('--step9_load_tactile_force6d_future', action='store_true',
                        help='Step9：返回 tactile force6d（raw 6D/hand）未来监督目标（默认关闭）')

    parser.add_argument('--step9_future_force_loss_weight', type=float, default=0.0,
                        help='Step9.3：未来力预测 MSE 辅助 loss 权重（0=关闭；建议对齐 Tavla 可用 0.1）')

    # ===== Step11.1：tactile_latent_shared -> future force aux loss（默认关闭） =====
    parser.add_argument(
        '--step11_future_force_loss_weight',
        type=float,
        default=0.0,
        help='Step11.1（默认关闭）：使用 tactile_latent_shared 的未来力预测辅助 loss 权重'
    )
    parser.add_argument(
        '--step11_tactile_pool',
        type=str,
        default='hand_mean_concat',
        choices=['hand_mean_concat', 'all_mean'],
        help="Step11.1 tactile pooling：'hand_mean_concat'(默认) 或 'all_mean'"
    )

    # ===== Step-9 (Phase B)：把力做成单 token，并选择注入位置（prefix/suffix） =====
    parser.add_argument(
        '--step9_force_token_inject',
        type=str,
        default='suffix',
        choices=['prefix', 'suffix'],
        help=(
            "Step9：把 /ft 和/或 tactile force6d 历史展平后编码成单 token，并注入到模型中。"
            "suffix=前置拼到 decoder queries；prefix=作为 encoder 输入的前缀 token（默认 suffix）"
        ),
    )

    # ===== Step-5.2（方向B）：encoder-layer 显式 cross-attn（默认仍 prepend，完全等价 Step-4） =====
    parser.add_argument(
        '--tactile_fusion',
        type=str,
        default='prepend',
        choices=['prepend', 'cross_v2t', 'cross_t2v', 'cross_bi'],
        help='tactile 融合方式：prepend(默认，等价Step-4)；cross_v2t(V2T)；cross_t2v(T2V)；cross_bi(双向)'
    )

    # ===== Step10.7 (P1.5): force tactile usage（默认关闭） =====
    parser.add_argument(
        '--use_tactile_residual_inject',
        action='store_true',
        help='(默认关闭) 将触觉 residual 注入到 vision 表示，结构性鼓励 post-contact 使用触觉'
    )
    parser.add_argument(
        '--tactile_residual_scale',
        type=float,
        default=1.0,
        help='触觉 residual 注入强度（仅在 --use_tactile_residual_inject 时生效；默认 1.0）'
    )
    parser.add_argument(
        '--vision_contact_noise_std',
        type=float,
        default=0.0,
        help=(
            '(默认 0=关闭) 仅 train-mode 且 post-contact 时对 vision 特征加小噪声，'
            '需要 dataset 在 tactile["_gate"] 里提供 t_onset/t_ref；建议 0.01~0.03'
        )
    )

    # ===== Step-6.1：导出 cross-attn 权重（仅在 cross_bi 下生效；不做 overlay） =====
    parser.add_argument(
        '--export_cross_attn',
        action='store_true',
        help='开启 cross-attn 导出（默认关闭；用于确保不加参数时行为完全回退）'
    )
    parser.add_argument(
        '--export_cross_attn_interval',
        type=int,
        default=0,
        help='每隔多少个 epoch 导出一次 cross-attn（默认0=关闭；如需每100个epoch导出一次则设为100）'
    )
    parser.add_argument(
        '--export_cross_attn_dir',
        type=str,
        default=None,
        help='cross-attn 导出目录（默认 ckpt_dir/cross_attn_export）'
    )
    parser.add_argument(
        '--export_cross_attn_layer_idx',
        type=int,
        nargs='+',
        default=[0],
        help=(
            '要导出的 encoder layer 索引（空格多值）。支持多层、-1=最后一层、-2=全部层。'
            '例如：--export_cross_attn_layer_idx 0 3 -1。'
            '若不包含 0，则不会导出第0层。默认: 0'
        )
    )

    # ===== 实时六维力（ft）可视化/保存（仅 real robot 生效） =====
    parser.add_argument('--visualize-ft', action='store_true', help='实时可视化六维力/矩 (ft)')
    parser.add_argument('--visualize-ft-hz', type=float, default=20.0, help='ft 曲线刷新频率 (Hz)')
    parser.add_argument('--visualize-ft-window', type=int, default=300, help='ft 曲线窗口（显示最近多少个 step）')
    parser.add_argument('--save-ft', action='store_true', help='保存每步 ft 数据（长度=episode 步数）到 .npy')
    parser.add_argument('--save-ft-path', type=str, default=None, help='保存 ft 的路径（默认写到 ckpt_dir/ft_rollout*.npy）')

    # ===== websocket eval mode（默认关闭） =====
    parser.add_argument('--ws_mode', type=str, default='off', choices=['off', 'server'],
                        help='eval 执行模式：off=本地 env.step；server=通过 websocket 发布 action，由机器人端执行')
    parser.add_argument('--ws_host', type=str, default='0.0.0.0',
                        help='ws_mode=server 时绑定的 host（默认 0.0.0.0）')
    parser.add_argument('--ws_port', type=int, default=8765,
                        help='ws_mode=server 时监听端口（默认 8765）')

    parsed_args = vars(parser.parse_args())
    exp_config_path = parsed_args.get('exp_config', None)
    if exp_config_path:
        _exp_cfg = load_exp_config(exp_config_path)
        parsed_args = apply_exp_config_to_args(parsed_args, _exp_cfg)
    main(parsed_args)
