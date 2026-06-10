#!/usr/bin/env python3
"""Look at the spyder model — live, or as a recorded clip.

  * live viewer (default) — interactive MuJoCo window: orbit/zoom with the mouse,
    drag a body to perturb. `--physics` also steps torques so you can watch it move.
  * --save OUT — headless offscreen render of a short rollout to GIF/MP4 (no
    window), for capturing a clip.

macOS: the live window must run under `mjpython` (it owns the Cocoa main thread):
    ./mujoco-env/bin/mjpython scripts/view.py
    ./mujoco-env/bin/mjpython scripts/view.py --physics --policy gentle
Headless recording runs under plain python:
    ./mujoco-env/bin/python scripts/view.py --save spyder_preview.gif
"""

from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import imageio
import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "scene.xml"
)


def _ctrl(policy: str, rng: np.random.Generator, nu: int) -> np.ndarray:
    """One control vector in the actuators' [-1, 1] range."""
    if policy == "zero":
        return np.zeros(nu, dtype=np.float32)
    if policy == "gentle":  # mild torques — easier to watch than full random
        return rng.uniform(-0.3, 0.3, nu).astype(np.float32)
    return rng.uniform(-1.0, 1.0, nu).astype(np.float32)  # "random"


def view(xml_path: str, physics: bool = False, policy: str = "gentle", seed: int = 0) -> None:
    """Open the interactive viewer; with `physics`, step torques so it moves.

    Uses launch_passive (a self-driven loop), not the managed launch(): the
    latter crashes on macOS ("Caught an unknown exception" in _Simulate).
    """
    model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if physics:
                data.ctrl[:] = _ctrl(policy, rng, model.nu)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep if physics else 1 / 60)


def record(
    xml_path: str,
    out: str = "spyder_preview.gif",
    steps: int = 300,
    policy: str = "gentle",
    seed: int = 0,
    fps: int = 30,
    every: int = 2,
    main_body: str = "torso",
    cam_azimuth: float | None = None,
    cam_elevation: float = -12.0,
    cam_distance: float = 3.0,
    size: int = 480,
) -> str:
    """Headless offscreen render of a rollout to GIF/MP4 — no GUI window.

    GIF needs no extra deps; .mp4 needs `pip install imageio-ffmpeg`.
    `main_body` is the torso/trunk body the Ant reward tracks ("trunk" for the quad).

    Camera: by default the Ant env's own tracking camera (a front three-quarter
    view). Pass `cam_azimuth` to instead render from an explicit orbit camera
    (azimuth/elevation/distance) that follows `main_body` — used to capture the
    same rollout from a second angle (e.g. a side profile) for a matched pair.
    """
    manual_cam = cam_azimuth is not None
    env = gym.make(
        "Ant-v5", xml_file=os.path.abspath(xml_path), main_body=main_body,
        render_mode=None if manual_cam else "rgb_array",
    )
    rng = np.random.default_rng(seed)
    nu = env.action_space.shape[0]
    env.reset(seed=seed)

    renderer = cam = body_id = None
    if manual_cam:
        # Render the env's own model/data with our camera, so physics + reset
        # logic stay identical to the default path — only the viewpoint changes.
        model, data = env.unwrapped.model, env.unwrapped.data
        renderer = mujoco.Renderer(model, size, size)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = cam_azimuth, cam_elevation, cam_distance
        body_id = model.body(main_body).id

    def _frame():
        if manual_cam:
            cam.lookat[:] = env.unwrapped.data.xpos[body_id]  # track the body as it moves
            renderer.update_scene(env.unwrapped.data, cam)
            return renderer.render()
        return env.render()

    frames = []
    for t in range(steps):
        _, _, terminated, truncated, _ = env.step(_ctrl(policy, rng, nu))
        if t % every == 0:
            frames.append(_frame())
        if terminated or truncated:
            env.reset()
    if renderer is not None:
        renderer.close()
    env.close()

    if out.lower().endswith(".gif"):
        imageio.mimsave(out, frames, duration=1000 * every / fps, loop=0)
    else:
        imageio.mimsave(out, frames, fps=fps // every)
    print(f"saved {len(frames)} frames -> {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xml", nargs="?", default=DEFAULT_MODEL, help="path to MJCF")
    p.add_argument("--physics", action="store_true", help="step torques in the live viewer")
    p.add_argument("--save", metavar="OUT", help="headless record to GIF/MP4 instead of a window")
    p.add_argument("--policy", choices=["zero", "random", "gentle"], default="gentle")
    p.add_argument("--steps", type=int, default=300, help="rollout length for --save")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--main-body", default="torso", help='reward body: "torso" (Ant) or "trunk" (quad)')
    p.add_argument("--cam-azimuth", type=float, default=None,
                   help="explicit orbit-camera azimuth for --save (else the env's tracking cam)")
    p.add_argument("--cam-elevation", type=float, default=-12.0)
    p.add_argument("--cam-distance", type=float, default=3.0)
    args = p.parse_args()

    if args.save:
        record(args.xml, out=args.save, steps=args.steps, policy=args.policy, seed=args.seed,
               main_body=args.main_body, cam_azimuth=args.cam_azimuth,
               cam_elevation=args.cam_elevation, cam_distance=args.cam_distance)
    else:
        view(args.xml, physics=args.physics, policy=args.policy, seed=args.seed)


if __name__ == "__main__":
    main()
