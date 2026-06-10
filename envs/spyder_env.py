#!/usr/bin/env python3
"""Register `Spyder-v0` — the 12-DoF tank quad as a Gymnasium env.

It reuses Gymnasium's generic `AntEnv` (forward-velocity reward, healthy-z
termination, ctrl/contact costs) — only the model and a few kwargs change, so
there is no bespoke RL code. This is *not* drop-in Ant-v5: the skeleton has 12
joints, so the obs/action shapes differ — which is exactly why it gets its own id.

Importing this module registers the env:
    import envs.spyder_env            # or: from envs import spyder_env
    env = gym.make("Spyder-v0")
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import spyder_params as P  # noqa: E402

SCENE = os.path.join(ROOT, "models", "scene_quad.xml")


def register_spyder() -> None:
    if "Spyder-v0" in gym.registry:
        return
    gym.register(
        id="Spyder-v0",
        entry_point="gymnasium.envs.mujoco.ant_v5:AntEnv",
        max_episode_steps=1000,
        kwargs=dict(
            xml_file=SCENE,
            main_body="trunk",          # forward-reward tracks the trunk
            healthy_z_range=P.HEALTHY_Z,  # re-tuned for the taller quad
            reset_noise_scale=0.05,
            frame_skip=5,               # dt = 0.05 s, same cadence as Ant
        ),
    )


register_spyder()
