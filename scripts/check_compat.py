#!/usr/bin/env python3
"""Prove a custom MJCF is still a drop-in Ant-v5.

Loads the model through `gym.make("Ant-v5", xml_file=...)` and asserts the v5
contract: observation Box(105,), action Box(8,), and the underlying skeleton of
1 free + 8 hinge joints, 13 bodies (excluding worldbody), 8 motors. Exits 0 on
success, 1 on any violation — so it doubles as this project's test.

Usage:
    ./mujoco-env/bin/python scripts/check_compat.py [path/to/model.xml]
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import mujoco

# The v5 contract. Changing the skeleton (Phase 4 / Spyder-v0) is the only
# legitimate reason for these to differ — and that gets its own env, not v5.
EXPECTED = {
    "obs": (105,),
    "action": (8,),
    "free_joints": 1,
    "hinge_joints": 8,
    "bodies": 13,  # excluding worldbody
    "motors": 8,
}

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "spyder.xml"
)


def inventory(model: mujoco.MjModel) -> dict:
    """Count skeleton elements straight from the compiled MjModel."""
    jnt_type = model.jnt_type
    return {
        "free_joints": int((jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum()),
        "hinge_joints": int((jnt_type == mujoco.mjtJoint.mjJNT_HINGE).sum()),
        "bodies": model.nbody - 1,  # body 0 is always the worldbody
        "motors": model.nu,
        "joint_names": [model.joint(i).name for i in range(model.njnt)],
    }


def check(xml_path: str) -> bool:
    xml_path = os.path.abspath(xml_path)
    if not os.path.exists(xml_path):
        print(f"✗ model not found: {xml_path}")
        return False

    env = gym.make("Ant-v5", xml_file=xml_path, main_body="torso")
    model = env.unwrapped.model

    actual = {
        "obs": env.observation_space.shape,
        "action": env.action_space.shape,
        **inventory(model),
    }
    env.close()

    print(f"model: {xml_path}")
    print(f"joints: {actual['joint_names']}\n")

    ok = True
    for key, want in EXPECTED.items():
        got = actual[key]
        mark = "✓" if got == want else "✗"
        if got != want:
            ok = False
        print(f"  {mark} {key:<13} expected {want!s:<8} got {got}")

    print("\nPASS — drop-in Ant-v5 ✓" if ok else "\nFAIL — v5 contract broken ✗")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    sys.exit(0 if check(path) else 1)
