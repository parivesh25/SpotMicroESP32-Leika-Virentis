"""Evaluate / watch a trained spot_pico residual policy in MuJoCo.

Loads the SB3 PPO actor + VecNormalize stats from a run dir and rolls out a command, either in
the interactive viewer or headless with tracking metrics.

Examples:
  python eval_policy.py runs/residual_pure_dr --vx 0.05
  python eval_policy.py runs/residual_pure_dr --headless
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.quadruped_mj_env import make_env
from src.sim.mj_runtime import CONTROL_DT


def load(run_dir):
    model_path = os.path.join(run_dir, "final_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(run_dir, "best", "best_model.zip")
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    model = PPO.load(model_path, device="cpu")
    return model, vn_path


def normalize_obs(vn, obs):
    return np.clip((obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--vx", type=float, default=0.05)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model, vn_path = load(args.run_dir)
    # VecNormalize stats for observation normalization (reward norm irrelevant at eval)
    vn = VecNormalize.load(vn_path, DummyVecEnv([make_env()]))
    env = make_env(seed=0)()
    env.fixed_command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    obs, _ = env.reset()
    n = int(args.seconds / CONTROL_DT)

    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.sim.model, env.sim.data)

    acc = np.zeros(3); cnt = 0
    for _ in range(n):
        a, _ = model.predict(normalize_obs(vn, obs), deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        acc += [info["fwd"], info["left"], info["yaw_rate"]]; cnt += 1
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
        if term or trunc:
            break
    if viewer is not None:
        viewer.close()
    v = acc / max(cnt, 1)
    print(f"cmd ({args.vx}, {args.vy}, {args.yaw}) -> mean fwd {v[0]:+.3f} left {v[1]:+.3f} "
          f"yaw {v[2]:+.2f}  (steps {cnt}, final height {env.sim.base_height():.4f})")


if __name__ == "__main__":
    main()
