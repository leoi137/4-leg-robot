#!/usr/bin/env python3
"""Register `Spyder-v0` — the 12-DoF ant-style spider as a Gymnasium env.

It reuses Gymnasium's generic `AntEnv` (forward-velocity reward, healthy-z
termination, ctrl/contact costs) — only the model and a few kwargs change, so
there is no bespoke RL code. This is *not* drop-in Ant-v5: each leg has a
third (lift) joint, so the action space is (12,) and the observation grows —
which is exactly why it gets its own id.

Spyder-v0 contract (see docs/design_spyder12.md):
  * 1 free root + 12 hinges (hip_i / lift_i / knee_i, legs 1-4), 13 bodies
  * 12 motors in per-leg order: hip_1, lift_1, knee_1, hip_2, ... knee_4
  * observation (113,) = qpos(17) + qvel(18) + cfrc_ext[1:] (13x6=78)

Importing this module registers the env:
    import envs.spyder_env            # or: from envs import spyder_env
    env = gym.make("Spyder-v0")
"""

from __future__ import annotations

import os

import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "models", "scene12.xml")

# Torso rests at z=0.35 and settles to ~0.33 (joint springs hold the arch).
# Terminate below 0.15 (belly-flop) — the upper bound stays generous so
# jumping doesn't end the episode.
HEALTHY_Z = (0.15, 1.0)


def register_spyder() -> None:
    if "Spyder-v0" in gym.registry:
        return
    gym.register(
        id="Spyder-v0",
        entry_point="gymnasium.envs.mujoco.ant_v5:AntEnv",
        max_episode_steps=1000,
        kwargs=dict(
            xml_file=SCENE,
            main_body="torso",           # forward-reward tracks the torso
            healthy_z_range=HEALTHY_Z,   # re-tuned for the low-slung stance
            reset_noise_scale=0.05,      # gentler than Ant's 0.1 — arched rest pose
            frame_skip=5,                # dt = 0.05 s, same cadence as Ant
        ),
    )


register_spyder()
