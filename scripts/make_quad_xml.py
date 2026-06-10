#!/usr/bin/env python3
"""Generate models/spyder_quad.xml — the 12-DoF tank quad MJCF — from spyder_params.

Generated (not hand-written) so the 12 joints, mesh placements, and collision
primitives can never drift from the single source of truth. Runs in mujoco-env.

Design:
- The default sprawl X-stance is BAKED into the body frames (each leg body carries
  the default joint rotation as its quat), so every joint's qpos0 = 0 IS the
  standing pose. A joint spring (springref 0) then holds it up at zero motor torque.
- Visual = mesh geoms + emissive amber accent primitives (class "visual", massless);
  collision = primitives (class "collision": boxes for hull/plates, foot spheres)
  that carry the mass.
- Accent lights are primitives, not meshes, so they can be tweaked without a CAD
  round-trip; they live on EXISTING bodies (never new bodies — the 13-body /
  12-joint contract is load-bearing).
- Collision density is solved so the whole robot weighs TARGET_MASS.

    ./mujoco-env/bin/python scripts/make_quad_xml.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spyder_params as P

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "spyder_quad.xml")

# Collision half-sizes. The legs collide as boxes matching the armor plates
# (they only ever meet the floor — self-collision is off); the foot stays a
# SPHERE: with no ankle joint the shank lands tilted, and a tilted box foot
# would balance on a corner edge (noisy contacts). Exported for the URDF script.
TRUNK_BOX = (P.TRUNK["length"] * 0.94 / 2, P.TRUNK["width"] * 0.94 / 2, P.TRUNK["height"] / 2)
HIP_BOX = (0.05, P.HIP_LEN, 0.05)  # half-sizes; gives the hip body mass (a joint body can't be massless)
THIGH_BOX = (P.THIGH_PLATE["w"] / 2, P.THIGH_PLATE["t"] / 2, P.THIGH_LEN / 2)
SHANK_BOX = (P.SHANK_BLOCK["w"] / 2 - 0.005, P.SHANK_BLOCK["t"] / 2 - 0.005, P.SHANK_LEN / 2)


def _quat_x(a):
    return (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)


def _quat_y(a):
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def _q(t):
    return " ".join(f"{v:.6f}" for v in t)


def _collision_density() -> float:
    """Density that makes (trunk box + 4*(hip + thigh + shank boxes + foot sphere)) = TARGET_MASS."""
    vbox = lambda h: 8 * h[0] * h[1] * h[2]
    sph = lambda r: (4 / 3) * math.pi * r ** 3
    legs = 4 * (vbox(HIP_BOX) + vbox(THIGH_BOX) + vbox(SHANK_BOX) + sph(P.FOOT_R))
    return P.TARGET_MASS / (vbox(TRUNK_BOX) + legs)


def _leg(name, sx, sy) -> str:
    haa, hfe, kfe = P.default_angles(sx, sy)
    hr = [math.radians(d) for d in P.HAA_RANGE_DEG]
    fr = [math.radians(d) for d in P.HFE_RANGE_DEG]
    kr = [math.radians(d) for d in P.KFE_RANGE_DEG]
    t = P.THIGH_PLATE["t"]
    return f"""
      <body name="{name}_hip" pos="{P.HIP_X*sx:.4f} {P.HIP_Y*sy:.4f} {P.HIP_Z:.4f}" quat="{_q(_quat_x(haa))}">
        <joint name="{name}_haa" axis="1 0 0" range="{hr[0]:.4f} {hr[1]:.4f}"/>
        <geom class="visual" mesh="hip" material="leg"/>
        <geom class="collision" type="box" size="{HIP_BOX[0]} {HIP_BOX[1]} {HIP_BOX[2]}"/>
        <body name="{name}_thigh" pos="0 {P.HIP_LEN*sy:.4f} 0" quat="{_q(_quat_y(hfe))}">
          <joint name="{name}_hfe" axis="0 1 0" range="{fr[0]:.4f} {fr[1]:.4f}"/>
          <geom class="visual" mesh="thigh" material="leg"/>
          <geom class="visual" type="box" size="0.025 0.004 0.008" pos="0 {sy*(t/2+0.003):.4f} -0.10" material="amber"/>
          <geom class="collision" type="box" size="{THIGH_BOX[0]:.4f} {THIGH_BOX[1]:.4f} {THIGH_BOX[2]:.4f}" pos="0 0 {-P.THIGH_LEN/2:.4f}"/>
          <body name="{name}_shank" pos="0 0 {-P.THIGH_LEN:.4f}" quat="{_q(_quat_y(kfe))}">
            <joint name="{name}_kfe" axis="0 1 0" range="{kr[0]:.4f} {kr[1]:.4f}"/>
            <geom class="visual" mesh="shank" material="leg"/>
            <geom class="collision" type="box" size="{SHANK_BOX[0]:.4f} {SHANK_BOX[1]:.4f} {SHANK_BOX[2]:.4f}" pos="0 0 {-P.SHANK_LEN/2:.4f}"/>
            <geom class="collision" type="sphere" pos="0 0 {-P.SHANK_LEN:.4f}" size="{P.FOOT_R}"/>
          </body>
        </body>
      </body>"""


def _accents() -> str:
    """Amber running lights + antennas on the trunk body (visual-only primitives)."""
    L, W = P.TRUNK["length"], P.TRUNK["width"]
    q45 = _q(_quat_y(math.radians(45)))
    out = [
        # glacis visor strip — sits proud in the slot make_meshes cut into the glacis
        f'<geom class="visual" type="box" size="0.006 0.16 0.010" pos="{L/2-0.046:.4f} 0 0.024" quat="{q45}" material="amber"/>',
        # turret visor on the turret's flat front face
        '<geom class="visual" type="box" size="0.004 0.10 0.008" pos="0.1210 0 0.0850" material="amber"/>',
    ]
    # two whip antennas on the turret roof
    for sy in (1, -1):
        out.append(f'<geom class="visual" type="cylinder" size="0.004 0.09" pos="-0.16 {sy*0.06:.3f} 0.24" material="leg"/>')
    # recessed running lights inside the side-skirt pocket groove (two per side)
    for sy in (1, -1):
        for sx in (1, -1):
            out.append(f'<geom class="visual" type="box" size="0.015 0.004 0.006" pos="{sx*0.10:.3f} {sy*0.210:.3f} -0.015" material="amber"/>')
    return "\n      ".join(out)


def build() -> str:
    h = P.standing_height()
    density = _collision_density()
    motors = "\n    ".join(
        f'<motor name="{j}" joint="{j}" gear="{P.GEAR}" ctrlrange="-1 1"/>' for j in P.JOINTS)
    legs = "".join(_leg(name, sx, sy) for name, sx, sy in P.LEGS)
    return f"""<mujoco model="spyder_quad">
  <compiler angle="radian" coordinate="local" meshdir="assets" autolimits="true"/>
  <option integrator="RK4" timestep="0.01"/>

  <default>
    <joint type="hinge" limited="true" damping="{P.JOINT_DAMPING}" armature="{P.JOINT_ARMATURE}" stiffness="{P.JOINT_STIFFNESS}"/>
    <geom condim="3" friction="1 0.5 0.5"/>
    <default class="visual">
      <geom type="mesh" group="2" contype="0" conaffinity="0" density="0"/>
    </default>
    <default class="collision">
      <geom group="3" contype="1" conaffinity="0" density="{density:.1f}"/>
    </default>
    <motor ctrllimited="true"/>
  </default>

  <asset>
    <mesh name="trunk" file="trunk.stl"/>
    <mesh name="hip" file="hip.stl"/>
    <mesh name="thigh" file="thigh.stl"/>
    <mesh name="shank" file="shank.stl"/>
    <mesh name="sensor" file="sensor.stl"/>
    <material name="hull" rgba="{_q(P.TRUNK_RGBA)}" specular="0.35" shininess="0.4"/>
    <material name="leg" rgba="{_q(P.LEG_RGBA)}" specular="0.3" shininess="0.35"/>
    <material name="amber" rgba="{_q(P.ACCENT_RGBA)}" emission="1.0"/>
  </asset>

  <worldbody>
    <body name="trunk" pos="0 0 {h:.4f}">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="0 -2.2 0.8" xyaxes="1 0 0 0 0.5 1"/>
      <geom class="visual" mesh="trunk" material="hull"/>
      <geom class="visual" mesh="sensor" pos="{P.SENSOR_POS[0]:.4f} {P.SENSOR_POS[1]:.4f} {P.SENSOR_POS[2]:.4f}" material="amber"/>
      {_accents()}
      <geom class="collision" type="box" size="{TRUNK_BOX[0]:.4f} {TRUNK_BOX[1]:.4f} {TRUNK_BOX[2]:.4f}"/>{legs}
    </body>
  </worldbody>

  <actuator>
    {motors}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    with open(OUT, "w") as f:
        f.write(build())
    print(f"wrote {OUT}  (standing height {P.standing_height():.3f} m, collision density {_collision_density():.0f} kg/m^3)")
