"""
Test script: send a single joint position from collected data to the robot.
Run on robot machine to verify robot moves correctly before full deployment.

Usage:
    python test_robot_single_step.py
    python test_robot_single_step.py --joint "-2.0,38.5,3.0,76.0,1.5,65.0,-16.0"
"""
import argparse
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser(description="Test robot single joint move")
    parser.add_argument("--joint", type=str, default=None,
                        help="Comma-separated 7D joint angles (degrees). "
                             "e.g. '-2.0,38.5,3.0,76.0,1.5,65.0,-16.0'")
    parser.add_argument("--action_mode", type=str, default="joint")
    args = parser.parse_args()

    from realman_env.envs.realman_env import RealmanEnv, Config

    cfg = Config()
    cfg.ACTION_MODE = args.action_mode
    env = RealmanEnv(cfg)

    # reset first
    print("[test] resetting robot...")
    obs = env.reset()
    current_joint = np.asarray(obs["proprio"], dtype=np.float32)
    print(f"[test] current joint: {current_joint}")

    if args.joint is None:
        print("[test] no --joint specified, only reset. Done.")
        return

    target = np.array([float(x) for x in args.joint.split(",")], dtype=np.float32)
    print(f"[test] target joint:  {target}")
    print(f"[test] diff:          {target - current_joint}")

    input("[test] press Enter to execute (Ctrl+C to cancel)...")

    obs, _ = env.step(target)
    new_joint = np.asarray(obs["proprio"], dtype=np.float32)
    print(f"[test] after step joint: {new_joint}")
    print(f"[test] error:            {np.abs(new_joint - target)}")
    print("[test] done.")


if __name__ == "__main__":
    main()
