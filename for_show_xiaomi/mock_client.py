"""
Mock Client - replays hdf5 dataset through the server for testing.

Usage:
    python -m for_show_xiaomi.mock_client \
        --host localhost \
        --dataset_dir /path/to/data \
        --episode_id 0

Verifies the full pipeline: TCP communication + model inference.
"""
from __future__ import annotations

import argparse
import logging

import h5py
import numpy as np

from for_show_xiaomi.tcp_client import (
    connect_to_server, recv_metadata,
    send_obs, recv_action,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="0 = full episode")
    parser.add_argument("--camera_names", type=str, default="global,wrist")
    parser.add_argument("--tac_side", type=str, default="left")
    parser.add_argument("--tac_key", type=str, default="img")
    parser.add_argument("--proprio_key", type=str, default="proprio_joint")
    parser.add_argument("--action_key", type=str, default="actions/joint_abs")
    args = parser.parse_args()

    cams = [s.strip() for s in args.camera_names.split(",") if s.strip()]

    # --- load episode ---
    path = f"{args.dataset_dir}/episode_{args.episode_id}.hdf5"
    print(f"[mock] loading {path}")

    with h5py.File(path, "r") as f:
        obs_g = f["observations"]

        images = {}
        for c in cams:
            key = f"images/{c}"
            if key in obs_g:
                images[c] = obs_g[key][()]
                print(f"  cam '{c}': {images[c].shape}")

        tac = None
        tac_path = f"tac/{args.tac_side}/{args.tac_key}"
        if tac_path in obs_g:
            tac = obs_g[tac_path][()]
            print(f"  tac: {tac.shape}")

        if args.proprio_key in obs_g:
            proprio = obs_g[args.proprio_key][()]
        elif "qpos" in obs_g:
            proprio = obs_g["qpos"][()]
        else:
            proprio = f[f"observations/{args.proprio_key}"][()]
        print(f"  proprio: {proprio.shape}")

        gt_actions = f[args.action_key][()]
        print(f"  gt actions: {gt_actions.shape}")

    T = gt_actions.shape[0]
    max_steps = args.max_steps if args.max_steps > 0 else T
    max_steps = min(max_steps, T)

    # --- connect ---
    sock = connect_to_server(args.host, args.port)
    errors = []

    try:
        meta = recv_metadata(sock)
        print(f"[mock] server metadata: {meta}")

        for t in range(max_steps):
            obs = {"images": {}, "qpos": proprio[t].astype(np.float32)}
            for c in cams:
                if c in images:
                    obs["images"][c] = images[c][t]
            if tac is not None:
                obs["tac"] = {args.tac_side: {args.tac_key: tac[t]}}

            send_obs(sock, obs)
            msg = recv_action(sock)

            pred = np.asarray(msg["actions"], dtype=np.float32)
            if pred.ndim >= 2:
                pred = pred[0]
            gt = gt_actions[t]
            err = np.abs(pred[:len(gt)] - gt[:len(pred)])
            errors.append(err)

            if t % 20 == 0 or t == max_steps - 1:
                print(f"  step {t:4d}  MAE={err.mean():.4f}  "
                      f"pred={np.array2string(pred, precision=3, suppress_small=True)}  "
                      f"gt={np.array2string(gt, precision=3, suppress_small=True)}")

    finally:
        sock.close()

    errors = np.array(errors)
    print(f"\n[mock] === Summary ===")
    print(f"  steps: {max_steps}")
    print(f"  per-dim MAE: {errors.mean(axis=0)}")
    print(f"  overall MAE: {errors.mean():.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
