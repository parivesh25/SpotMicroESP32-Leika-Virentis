# spot_pico simulation — MuJoCo residual-gait training

MuJoCo physics + Gymnasium + Stable-Baselines3 PPO for training a **residual walking policy**
on the `spot_pico` quadruped. The baseline is the ESP32 firmware gait ported to NumPy; the
policy learns only small per-foot corrections on top (`residual_pure`, à la spot_mini_mini D2 /
the Hexapod project). A zero action reproduces the firmware gait, so training starts from a
working gait and only learns stabilization.

Managed with **uv** (Python ≥ 3.13).

```bash
uv sync
```

## Architecture

```
command [vx, vy, yaw]  ->  analytic_gait_action  ->  firmware gait (phase clock, trot
                                                     offsets, stance + 12-pt Bezier swing)
   -> Cartesian foot targets (base frame)  + policy residual (12-D, ±15 mm)
   -> analytic IK  ->  12 joint angles  ->  MuJoCo position actuators (100 Hz control)
```

- `src/robot/firmware_gait.py` — NumPy port of `esp32/.../walk_state.h` + spot_pico analytic IK
  (verified to 0.0 mm against MuJoCo forward kinematics) + command→gait map.
- `src/sim/mj_runtime.py` — MuJoCo runtime wrapper (load model, IK→ctrl, IMU/contacts).
- `src/sim/domain_rand.py` — per-episode/step domain randomization for sim-to-real.
- `src/envs/quadruped_mj_env.py` — Gymnasium env; 12-D residual action, 38-D hardware-only
  observation (gravity-in-body, gyro, rpy, prev joint cmd, gait phase, command, prev action),
  ANYmal-style exponential velocity/yaw tracking reward.
- `src/resources/spot_pico/` — MJCF `scene.xml`, meshes, `linkage.py` (four-bar tibia map,
  deploy-time only), tuned `gait_coef.json`.
- `src/leika/` — high-level `Robot` façade (`stand`/`walk`/`rest`) over the sim.

The robot model comes from `spot_pico_description`. The four-bar tibia linkage (`linkage.py`)
is only needed when deploying to real servos; in sim the URDF tibia joint is actuated directly.

## Commands

```bash
uv run python -m src.robot.firmware_gait   # IK/FK self-test + stance angles
uv run python replay_gait.py --headless    # validate the zero-residual gait (all directions)
uv run python replay_gait.py --vx 0.05     # watch the baseline gait in the viewer

uv run python optimize_gait.py --iters 30  # calibrate command->gait map -> gait_coef.json

uv run python train_mj.py --smoke                                   # pipeline sanity run
uv run python train_mj.py --randomize --zero-final --init-std 0.3 \
    --curriculum --resample-steps 300 --timesteps 5_000_000 --num-envs 16

uv run python eval_policy.py runs/residual_pure_dr --vx 0.05        # watch a trained policy
uv run python visionary_demo.py                                     # high-level API demo

uv run pytest -q                                                    # regression tests
```

Uneven ground: pass `--terrain 0.008` to `train_mj.py` to train on the heightfield scene
(`scene_terrain.xml`) with per-episode random bumps up to the given height in metres. The policy
is blind (IMU only, no exteroception), so terrain only enters through proprioception.

TensorBoard: `uv run tensorboard --logdir runs`. Watch `terms/r_vel` (tracking) rise and
`terms/p_res` (residual magnitude) stay small — the policy should stabilize, not replace, the gait.

## Follow-ups (out of scope here)

- **Sim-to-real export**: bake actor + VecNormalize stats + analytic gait map + `servo_from_tibia`
  into a C++ header for the ESP32 (see Hexapod's `export_policy.py`).
- Hardware validation of IK joint-sign conventions and the four-bar linkage.
