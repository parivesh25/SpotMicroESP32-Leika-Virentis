"""High-level "visionary" façade over the MuJoCo spot_pico simulation.

Runs the ported firmware gait in a background thread stepping SpotPicoSim at the control rate,
and maps semantic commands (stand / walk / rest) onto the [vx, vy, yaw] command consumed by the
analytic gait. This is a thin convenience layer; for visualization/training use replay_gait.py,
eval_policy.py, or train_mj.py.
"""

import threading
import time

import numpy as np

from ..sim.mj_runtime import SpotPicoSim, CONTROL_DT
from ..robot.firmware_gait import GaitController, GaitState, BodyState, analytic_gait_action, set_mode, TROT
from .constants import Gait, Mode

# walk() argument (fraction, -1..1) -> velocity command scaling
MAX_VX = 0.06   # m/s forward
MAX_VY = 0.03   # m/s lateral
MAX_YAW = 2.0   # rad/s


class Robot:
    def __init__(self, target: str = "simulation"):
        if target != "simulation":
            raise NotImplementedError(f"target {target!r} not supported yet in the visionary API.")
        self.target = target
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self.sim = SpotPicoSim()
        self.gait = GaitState()
        set_mode(self.gait, TROT)
        self.gc = GaitController()
        self.body = BodyState()
        self.mode = Mode.REST
        self._cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, yaw]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        self.sim.reset_to_stand()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[*] Leika Robot connected to {self.target}")

    def disconnect(self):
        print("[*] Disconnecting and putting robot to rest...")
        self.rest()
        time.sleep(0.5)
        self._running = False
        if self._thread:
            self._thread.join()
        print("[+] Disconnected.")

    def _run_loop(self):
        while self._running:
            start = time.time()
            with self._lock:
                cmd = self._cmd.copy()
            analytic_gait_action(cmd, self.gait)
            self.gc.advance_phase(self.gait, CONTROL_DT)
            self.gc.generate_feet(self.gait, self.body)
            self.sim.set_joint_targets(self.sim.body_targets_from_feet(self.body))
            self.sim.step_physics()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - start)))

    def _set_cmd(self, vx, vy, yaw):
        with self._lock:
            self._cmd[:] = [vx, vy, yaw]

    # High-level API ---------------------------------------------------------
    def stand(self, height: float = 0.5):
        self.mode = Mode.STAND
        self._set_cmd(0.0, 0.0, 0.0)  # hold the nominal stance (no gait)

    def rest(self):
        self.mode = Mode.REST
        self._set_cmd(0.0, 0.0, 0.0)

    def walk(self, x: float = 0.0, y: float = 0.0, turn: float = 0.0, speed: float = 1.0):
        self.mode = Mode.WALK
        s = float(np.clip(speed, 0.0, 1.0)) if speed <= 1.0 else 1.0
        vx = float(np.clip(x, -1, 1)) * MAX_VX
        vy = float(np.clip(y, -1, 1)) * MAX_VY
        yaw = float(np.clip(turn, -1, 1)) * MAX_YAW
        self._set_cmd(vx, vy, yaw)

    @property
    def pose(self):
        return self

    @property
    def position(self):
        a = self.sim.base_qposadr
        return tuple(float(v) for v in self.sim.data.qpos[a : a + 3])
