from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import websockets.sync.client
from tools.websocket import msgpack_numpy

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from tools.websocket import msgpack_numpy

def _expected_action_dim(action_mode: str) -> int:
    if action_mode == "joint":
        return 7
    if action_mode == "eef_rel":
        return 6
    raise ValueError(f"Unsupported action_mode={action_mode!r}")


def _wait_for_server(uri: str):
    logging.info("[miact-ws-client] waiting for server at %s ...", uri)
    while True:
        try:
            return websockets.sync.client.connect(
                uri,
                compression=None,
                max_size=None,
            )
        except ConnectionRefusedError:
            logging.info("[miact-ws-client] server not ready, retry in 5s")
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIACT Realman websocket client",
    )
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--action_mode",
        type=str,
        choices=["joint", "eef_rel"],
        default="joint",
    )
    parser.add_argument("--control_hz", type=float, default=20.0)
    parser.add_argument("--max_steps", type=int, default=250)
    args = parser.parse_args()

    from realman_env.envs.realman_env import (
        RealmanEnv,
        Config as RealmanEnvConfig,
    )  # local import

    env_cfg = RealmanEnvConfig()
    env_cfg.ACTION_MODE = args.action_mode
    # Step13/Step9 force-aware models require /ft to be present in obs.
    # RealmanEnv only includes obs['ft'] when SAVE_FT=True.
    env_cfg.SAVE_FT = True
    env = RealmanEnv(env_cfg)

    uri = f"ws://{args.host}:{args.port}"
    logging.info("[miact-ws-client] connecting to %s", uri)

    packer = msgpack_numpy.Packer()
    dt = 1.0 / float(args.control_hz)
    action_dim = _expected_action_dim(args.action_mode)

    with _wait_for_server(uri) as conn:
        raw_meta = conn.recv()
        if isinstance(raw_meta, str):
            raise RuntimeError(
                f"Expected binary metadata, got text: {raw_meta}",
            )
        metadata = msgpack_numpy.unpackb(raw_meta)
        logging.info("[miact-ws-client] server metadata: %s", metadata)

        obs = env.reset()
        obs["eef_pose"] = env.current_pose.copy()  # 6D: x,y,z,rx,ry,rz

        max_steps = int(args.max_steps)
        for step_idx in range(max_steps):
            t0 = time.perf_counter()
            conn.send(packer.pack(obs))
            raw = conn.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Error in inference server:\n{raw}")
            msg = msgpack_numpy.unpackb(raw)
            actions = np.asarray(msg.get("actions"), dtype=np.float32)
            if actions.ndim == 1:
                action = actions.reshape(-1)
            elif actions.ndim >= 2:
                action = actions[0].reshape(-1)
            else:
                raise RuntimeError(
                    "Expected 'actions' with shape (H, D), got "
                    f"{actions.shape}",
                )
            if action.shape[0] != action_dim:
                raise ValueError(
                    f"Expected action_dim={action_dim}, "
                    f"got shape={action.shape}",
                )

            obs, _rew = env.step(action)
            obs["eef_pose"] = env.current_pose.copy()  # 6D: x,y,z,rx,ry,rz

            # MITA-VLA-like progress logging
            logging.info(
                "[miact-ws-client] step %d/%d",
                step_idx + 1,
                max_steps,
            )

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
