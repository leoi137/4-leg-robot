#!/usr/bin/env python3
"""Preview the spyder tank meshes in MuJoCo (runs in the 3.14 mujoco-env).

Loads the STLs authored by `make_meshes.py` (cad-env) and renders: each part on
its own, plus a "hero" shot of the whole robot posed in the default ∩ stance via
`spyder_params` forward-kinematics — all before any physics XML exists.

    ./mujoco-env/bin/python scripts/render_robot.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio
import mujoco
import numpy as np

import spyder_params as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "models", "assets")


def _mat2quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, float).reshape(9))
    return tuple(q)


def mesh(name):
    return os.path.join(ASSETS, f"{name}.stl")


def render(items, out, floor=False, px=720, dist=1.0, az=130, el=-20, lookat=(0, 0, 0)):
    """Render placed meshes. items: (mesh_path, pos, quat(wxyz), rgba)."""
    assets, geoms = "", ""
    for i, (path, pos, quat, rgba) in enumerate(items):
        assets += f'<mesh name="m{i}" file="{path}"/>'
        geoms += (f'<geom type="mesh" mesh="m{i}" pos="{pos[0]} {pos[1]} {pos[2]}" '
                  f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" '
                  f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>')
    world = ""
    if floor:
        assets += ('<texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.6 0.7" rgb2="0.1 0.1 0.15" width="64" height="64"/>'
                   '<texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.19 0.22" rgb2="0.24 0.25 0.29" width="128" height="128"/>'
                   '<material name="grid" texture="grid" texrepeat="12 12" reflectance="0.1"/>')
        world += '<geom name="floor" type="plane" size="6 6 0.1" material="grid"/>'
    xml = (f'<mujoco><visual><global offwidth="{px}" offheight="{px}"/>'
           f'<headlight diffuse=".6 .6 .6" ambient=".35 .35 .35" specular="0.1 0.1 0.1"/><quality shadowsize="4096"/></visual>'
           f'<asset>{assets}</asset>'
           f'<worldbody><light pos="1 -1 2.2" dir="-0.4 0.4 -1" directional="true" diffuse="0.7 0.7 0.7"/>'
           f'{world}{geoms}</worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    cam.lookat[:] = lookat
    r = mujoco.Renderer(m, px, px)
    r.update_scene(d, cam)
    imageio.imwrite(out, r.render())


def render_parts():
    gun, leg, acc = P.TRUNK_RGBA, P.LEG_RGBA, P.ACCENT_RGBA
    Q = (1, 0, 0, 0)
    render([(mesh("trunk"), (0, 0, 0), Q, gun),
            (mesh("sensor"), P.SENSOR_POS, Q, acc)],
           "/tmp/part_trunk.png", dist=1.25, lookat=(0, 0, 0.05))
    render([(mesh("hip"), (-0.35, 0, 0), Q, leg),
            (mesh("thigh"), (0.0, 0, 0.13), Q, leg),
            (mesh("shank"), (0.35, 0, 0.14), Q, leg)],
           "/tmp/part_legs.png", dist=1.15, az=110, el=-10, lookat=(0, 0, -0.02))
    print("rendered /tmp/part_trunk.png and /tmp/part_legs.png")


def render_hero(out="/tmp/hero.png"):
    gun, leg, acc = P.TRUNK_RGBA, P.LEG_RGBA, P.ACCENT_RGBA
    h = P.standing_height()
    off = np.array([0, 0, h])
    sensor = np.array(P.SENSOR_POS) + off
    items = [(mesh("trunk"), (0, 0, h), (1, 0, 0, 0), gun),
             (mesh("sensor"), tuple(sensor), (1, 0, 0, 0), acc)]
    for name, sx, sy in P.LEGS:
        fk = P.forward_kinematics(sx, sy, *P.default_angles(sx, sy))
        items.append((mesh("hip"), tuple(fk["hip"] + off), _mat2quat(fk["r_hip"]), leg))
        items.append((mesh("thigh"), tuple(fk["hfe"] + off), _mat2quat(fk["r_thigh"]), leg))
        items.append((mesh("shank"), tuple(fk["knee"] + off), _mat2quat(fk["r_shank"]), leg))
    # low front three-quarter, like the reference shot
    render(items, out, floor=True, dist=2.4, az=160, el=-8, lookat=(0, 0, 0.20))
    print(f"rendered {out}")


if __name__ == "__main__":
    render_parts()
    render_hero()
