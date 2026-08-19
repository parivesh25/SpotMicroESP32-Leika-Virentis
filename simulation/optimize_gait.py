"""Calibrate the analytic command->gait map so the ZERO-RESIDUAL firmware gait roughly tracks
commanded velocity in MuJoCo. Black-box (differential evolution) tuning of the GAIT_COEF gains;
the result is written to src/resources/spot_pico/gait_coef.json and loaded by firmware_gait.

This is the spot_pico analogue of Hexapod's optimize_gait.py. A well-calibrated baseline is what
makes residual_pure work: a zero policy already tracks velocity, so the policy only stabilizes.

  python optimize_gait.py --iters 30
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import differential_evolution

from src.sim.mj_runtime import SpotPicoSim, CONTROL_DT
from src.robot import firmware_gait as fg
from src.envs.quadruped_mj_env import _rot_vec_by_conj

# test commands (robot frame): forward speeds, lateral, yaw
TEST_CMDS = [(0.03, 0, 0), (0.05, 0, 0), (0.07, 0, 0), (-0.03, 0, 0),
             (0, 0.03, 0), (0, -0.03, 0), (0, 0, 1.0), (0, 0, -1.0)]
PARAM_NAMES = ["gain_x", "gain_y", "gain_yaw", "speed_base", "speed_slope"]
BOUNDS = [(0.02, 0.15), (0.02, 0.15), (0.7, 6.0), (0.2, 0.7), (0.3, 2.5)]


def mean_velocity(sim, coef, cmd, seconds=4.0, warmup=1.0):
    """Roll out the zero-residual gait; return mean (fwd, left, yaw_rate) over the post-warmup window."""
    fg.GAIT_COEF.update(coef)
    gait = fg.GaitState(); fg.set_mode(gait, fg.TROT)
    gc = fg.GaitController(); body = fg.BodyState()
    sim.reset_to_stand()
    n, w = int(seconds / CONTROL_DT), int(warmup / CONTROL_DT)
    acc = np.zeros(3); cnt = 0
    for t in range(n):
        fg.analytic_gait_action(cmd, gait)
        gc.advance_phase(gait, CONTROL_DT)
        gc.generate_feet(gait, body)
        sim.set_joint_targets(sim.body_targets_from_feet(body))
        sim.step_physics()
        if t >= w:
            q = sim.base_quat()
            vb = _rot_vec_by_conj(q, sim.data.qvel[0:3].copy())
            acc += [-vb[1], vb[0], sim.gyro()[2]]
            cnt += 1
        if sim.base_height() < 0.03:
            return None  # fell -> invalid
    return acc / max(cnt, 1)


def objective(x, sim):
    coef = dict(zip(PARAM_NAMES, x))
    err = 0.0
    for cmd in TEST_CMDS:
        v = mean_velocity(sim, coef, cmd)
        if v is None:
            return 1e3
        # normalize yaw error to comparable scale (rad/s vs m/s)
        e = np.array([v[0] - cmd[0], v[1] - cmd[1], (v[2] - cmd[2]) * 0.03])
        err += float(e @ e)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30, help="DE max iterations")
    ap.add_argument("--popsize", type=int, default=12)
    args = ap.parse_args()

    sim = SpotPicoSim()
    print("initial objective:", round(objective([fg.GAIT_COEF[n] for n in PARAM_NAMES], sim), 5))

    result = differential_evolution(
        objective, BOUNDS, args=(sim,), maxiter=args.iters, popsize=args.popsize,
        tol=1e-4, seed=0, polish=True, disp=True,
    )
    best = dict(zip(PARAM_NAMES, result.x))
    coef = dict(fg.GAIT_COEF); coef.update(best)
    out = os.path.join(os.path.dirname(__file__), "src", "resources", "spot_pico", "gait_coef.json")
    with open(out, "w") as f:
        json.dump(coef, f, indent=2)
    print(f"\nbest objective {result.fun:.5f}\nbest params {best}\nwrote {out}")

    fg.GAIT_COEF.update(best)
    print("\nachieved vs commanded:")
    for cmd in TEST_CMDS:
        v = mean_velocity(sim, best, cmd)
        print(f"  cmd {cmd} -> fwd {v[0]:+.3f} left {v[1]:+.3f} yaw {v[2]:+.2f}")


if __name__ == "__main__":
    main()
