"""
TactileACT TCP Client for Realman Robot (runs on robot machine)

Usage:
    python ws_client.py --host <gpu_server_ip> --port 8765

Prerequisites on robot machine:
    - realman_env package (from miACT project)
    - Robotic_Arm SDK
    - tcp_client.py (same directory)

The obs dict sent to server must match serve_policy.py expectations:
    {
        "images": {"global": (H,W,3) uint8, "wrist": (H,W,3) uint8},
        "tac":    {"left": {"img": (240,240,3) uint8}},
        "qpos":   (state_dim,) float32,
    }
"""
from __future__ import annotations
elf.env = RealmanEnv(cfg) 
import argparse
import logging
import time

import numpy as np

from tcp_client import (
    connect_to_server, recv_metadata,
    send_obs, recv_action,
)


class RobotEnv:
    """Wraps miACT's RealmanEnv, adapting obs format for TactileACT server."""

    def __init__(self, action_mode: str = "joint"):
        from realman_env.envs.realman_env  importRealmanEnv, Config

        cfg = Config()
        cfg.ACTION_MODE = action_mode
        self.env = RealmanEnv(cfg)

    def reset(self) -> dict:
        """Move to home, return first obs."""
        obs = self.env.reset()
        return self._build_obs(obs)

    def step(self, action: np.ndarray) -> dict:
        """Execute action, return new obs."""
        obs, _ = self.env.step(action)
        return self._build_obs(obs)

    def _build_obs(self, raw_obs: dict) -> dict:
        """Convert RealmanEnv obs to TactileACT server format.

        RealmanEnv produces:
            raw_obs["images"][cam]          -> (H,W,3) uint8
            raw_obs["proprio"]              -> (7,) float32 (joint deg)
            raw_obs["tactile"][side]["img"]  -> (240,240,3) uint8

        Server expects:
            obs["images"][cam]    -> (H,W,3) uint8
            obs["qpos"]          -> (state_dim,) float32
            obs["tac"][side]["img"] -> (240,240,3) uint8
        """
        obs = {
            "images": raw_obs.get("images", {}),
            "qpos": np.asarray(raw_obs["proprio"], dtype=np.float32),
        }

        # tactile: rename "tactile" -> "tac"
        tactile = raw_obs.get("tactile")
        if tactile is not None:
            tac = {}
            for side, side_data in tactile.items():
                if isinstance(side_data, dict) and "img" in side_data:
                    tac[side] = {"img": side_data["img"]}
            if tac:
                obs["tac"] = tac

        return obs


def main():
    parser = argparse.ArgumentParser(
        description="TactileACT Realman client (runs on robot machine)",
    )
    parser.add_argument("--host", type=str, required=True,
                        help="GPU server IP address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--control_hz", type=float, default=20.0)
    parser.add_argument("--action_mode", type=str, default="joint",
                        choices=["joint", "eef_rel"])
    args = parser.parse_args()

    env = RobotEnv(action_mode=args.action_mode) #创建机器人环境
    dt = 1.0 / args.control_hz

    sock = connect_to_server(args.host, args.port)  #TCP连接GPU推理server
    try:
        metadata = recv_metadata(sock) #接受GPU配置信息
        logging.info("[client] server metadata: %s", metadata)
        #机器人复位初始位置
        obs = env.reset() 

        for step in range(args.max_steps): #默认300
            t0 = time.perf_counter()

            send_obs(sock, obs)
            msg = recv_action(sock)

            action = np.asarray(msg["actions"], dtype=np.float32)
            if action.ndim >= 2:
                action = action[0]

            obs = env.step(action)

            logging.info("[client] step %d/%d", step + 1, args.max_steps)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        sock.close()
    print("[client] done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
下