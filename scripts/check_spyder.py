#!/usr/bin/env python3
"""Verify the `Spyder-v0` contract — the 12-DoF analogue of check_compat.py.

Asserts the skeleton (1 free + 12 hinge, 12 motors, 13 bodies), that the env
loads with obs/action shapes matching the compiled model, and that the robot
actually STANDS: a zero-torque rollout must survive well past instant
termination (mean episode length high => it holds its stance). Exits 0/1.

    ./mujoco-env/bin/python scripts/check_spyder.py
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "envs"))
import spyder_env  # noqa: E402,F401  (registers Spyder-v0)

EXPECTED = {"action": (12,), "free_joints": 1, "hinge_joints": 12, "bodies": 13, "motors": 12}
STAND_MIN_STEPS = 200  # of a 1000-step episode; zero-torque, must not topple early


def inventory(model: mujoco.MjModel) -> dict:
    jt = model.jnt_type
    return {
        "free_joints": int((jt == mujoco.mjtJoint.mjJNT_FREE).sum()),
        "hinge_joints": int((jt == mujoco.mjtJoint.mjJNT_HINGE).sum()),
        "bodies": model.nbody - 1,
        "motors": model.nu,
    }


def standing_steps(env, episodes=3) -> float:
    """Mean steps survived under zero torque (springs should hold the stance)."""
    lengths = []
    for ep in range(episodes):
        env.reset(seed=ep)
        n = 0
        for _ in range(1000):
            _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape[0], np.float32))
            n += 1
            if terminated or truncated:
                break
        lengths.append(n)
    return float(np.mean(lengths))


def main() -> int:
    env = gym.make("Spyder-v0")
    model = env.unwrapped.model
    actual = {"action": env.action_space.shape, **inventory(model)}

    print(f"obs space: {env.observation_space.shape}   action space: {env.action_space.shape}")
    ok = True
    for key, want in EXPECTED.items():
        got = actual[key]
        mark = "✓" if got == want else "✗"
        ok &= got == want
        print(f"  {mark} {key:<13} expected {want!s:<8} got {got}")

    mean_len = standing_steps(env)
    stands = mean_len >= STAND_MIN_STEPS
    ok &= stands
    print(f"  {'✓' if stands else '✗'} stands         zero-torque mean episode length {mean_len:.0f} "
          f"(need >= {STAND_MIN_STEPS})")
    env.close()

    print("\nPASS — Spyder-v0 contract holds ✓" if ok else "\nFAIL — contract broken ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
