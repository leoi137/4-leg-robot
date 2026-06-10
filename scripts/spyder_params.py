#!/usr/bin/env python3
"""Single source of truth for the 12-DoF spyder "tank" quad (Spyder-v0).

Every downstream artifact imports from here so the build123d meshes, the MuJoCo
XML, and the URDF can never disagree on dimensions, joint axes, or the default
sprawl X-stance (legs splayed wide like a spider, front knees raked forward
and rear knees raked back — the silhouette of docs/media/4_leg_robot.png).

Conventions
-----------
- Units: metres and radians (constants ending `_DEG` are degrees, converted on use).
- Trunk frame (MuJoCo): +x forward, +y left, +z up.
- Each leg = 3 joints: HAA (hip, swings leg sideways, axis +x)
                       HFE (hip, swings thigh fore/aft, axis +y)
                       KFE (knee, bends shank, axis +y).
- Mesh local frames (so a part drops straight into its body):
    trunk  -> centred on the origin, sized TRUNK.
    hip    -> centred on the origin (HAA pod, roughly symmetric so it needs no mirror).
    thigh  -> hangs along -z, length THIGH_LEN, top at the origin.
    shank  -> hangs along -z, length SHANK_LEN, top at the origin, foot cap at the bottom.

Run it (`python scripts/spyder_params.py`) to print a sanity summary.
"""

from __future__ import annotations

import math

import numpy as np

# ---- trunk (the lower tank hull): full sizes in x (length), y (width), z (height) ----
# The turret deck rides on top as extra *visual* height (see TURRET below).
TRUNK = dict(length=0.64, width=0.44, height=0.14)

# ---- leg segment lengths ----
HIP_LEN = 0.09     # HAA pod: lateral offset from the HAA axis out to the HFE axis
THIGH_LEN = 0.22   # HFE axis -> KFE axis
SHANK_LEN = 0.26   # KFE axis -> foot contact
FOOT_R = 0.04      # foot sphere radius (the actual contact geometry)
LEG_R = 0.04       # leg segment thickness reference

# ---- where each leg attaches to the trunk (trunk frame, lower hull corners) ----
HIP_X = 0.24       # fore/aft offset of the hips from trunk centre
HIP_Y = 0.19       # left/right offset
HIP_Z = -0.02      # hips sit just below the trunk centreline

# legs: (name, sign_x [front +1 / back -1], sign_y [left +1 / right -1])
LEGS = [("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)]

# ---- joint axes, expressed in each joint's parent frame ----
HAA_AXIS = (1, 0, 0)
HFE_AXIS = (0, 1, 0)
KFE_AXIS = (0, 1, 0)

# ---- joint limits (degrees), RELATIVE to the default stance ----
# The XML bakes the ∩ stance into the body frames, so each joint's 0 IS the
# default angle below; these ranges are how far it may travel either side of it.
HAA_RANGE_DEG = (-35, 35)
HFE_RANGE_DEG = (-70, 70)
KFE_RANGE_DEG = (-70, 70)

# ---- default sprawl X-stance (degrees): wide, low, spider-planted ----
# A large HAA roll cants each leg's whole bend plane outboard, so the knee rides
# high and outside the hull and the shank dives back down-and-in — the spider
# silhouette. HFE/KFE are mirrored front/back (X-stance): front knees rake
# forward, rear knees rake back, so the legs reach away from the body in all
# four directions. Foot height only depends on cos() of the angles (even
# function), so the mirroring keeps all four feet at the same z.
HAA_DEF_DEG = 45    # the sprawl knob: outboard splay (footprint ~2.5x hull width)
HFE_DEF_DEG = 25    # thigh rake from vertical (sign flipped per front/back pair)
KFE_DEF_DEG = -50   # knee fold so the shank drops back toward the ground

# ---- look ----
TRUNK_RGBA = (0.08, 0.085, 0.10, 1.0)  # dark gunmetal armor
LEG_RGBA = (0.13, 0.14, 0.16, 1.0)     # graphite plate
ACCENT_RGBA = (1.0, 0.62, 0.18, 1.0)   # amber running lights / visor

# ---- shared visual dims (CAD meshes + XML accents both read these; no drift) ----
THIGH_PLATE = dict(w=0.20, t=0.065, top_ext=0.05)  # x-width, y-thickness, rise above HFE origin
SHANK_BLOCK = dict(w=0.115, t=0.075)               # x-width, y-thickness of the lower-leg block
FOOT_PAD = dict(x=0.14, y=0.11, h=0.035)           # broad visual foot pad (contact stays a sphere)
TURRET = dict(length=0.34, width=0.26, height=0.085)  # raised upper deck on the hull
TURRET_POS = (-0.05, 0.0, 0.1075)   # turret centre in the trunk frame (z = H/2 + Th/2 - 0.005)
SENSOR_POS = (0.025, 0.0, 0.1725)   # optic dome, half-embedded in the sensor block's front

# ---- physics ----
TARGET_MASS = 6.0      # kg, whole robot; collision density is solved to hit this
GEAR = 40              # motor gear (torque scale)
JOINT_DAMPING = 3.0    # velocity damping per joint (keeps damping ratio ~0.5 at k=60)
JOINT_ARMATURE = 0.10  # reflected rotor inertia (stabilises the integrator)
# Spring toward the rest stance (joint=0) so it stands at zero torque. The sprawl
# puts each foot ~0.37 m laterally off its HAA axis, so gravity loads each hip
# with ~5.5 N·m; sag = torque/stiffness, and k=60 keeps that to ~5° (~3.5 cm of
# trunk drop). The old k=12 would let the legs splat out ~26° and fail HEALTHY_Z.
JOINT_STIFFNESS = 60
HEALTHY_Z = (0.15, 0.45)  # trunk z must stay here or the episode terminates (standing ~0.304)

# Actuator order = leg order x [HAA, HFE, KFE]; positional, like Ant's motors.
JOINTS = [f"{name}_{j}" for name, _, _ in LEGS for j in ("haa", "hfe", "kfe")]


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def default_angles(sx: int, sy: int) -> tuple[float, float, float]:
    """(haa, hfe, kfe) in radians for the leg with the given front/left signs."""
    haa = math.radians(HAA_DEF_DEG) * sy     # splay outward on each side (mirror L/R)
    hfe = math.radians(HFE_DEF_DEG) * (-sx)  # X-stance: front knees rake fwd, rear back
    kfe = math.radians(KFE_DEF_DEG) * (-sx)
    return haa, hfe, kfe


def forward_kinematics(sx: int, sy: int, haa: float, hfe: float, kfe: float) -> dict:
    """Key points + frames of one leg in the trunk frame (for previews + init height)."""
    hip = np.array([HIP_X * sx, HIP_Y * sy, HIP_Z])
    r_hip = _rot_x(haa)
    hfe_pt = hip + r_hip @ np.array([0.0, HIP_LEN * sy, 0.0])
    r_thigh = r_hip @ _rot_y(hfe)
    knee = hfe_pt + r_thigh @ np.array([0.0, 0.0, -THIGH_LEN])
    r_shank = r_thigh @ _rot_y(kfe)
    foot = knee + r_shank @ np.array([0.0, 0.0, -SHANK_LEN])
    return dict(hip=hip, hfe=hfe_pt, knee=knee, foot=foot,
                r_hip=r_hip, r_thigh=r_thigh, r_shank=r_shank)


def standing_height() -> float:
    """Trunk-origin height that just lands the lowest foot SURFACE on z=0.

    The FK foot point is the centre of the FOOT_R contact sphere, so we add
    FOOT_R: contact happens at the sphere's surface, not its frame origin.
    (Without this the feet start buried 4 cm in the floor and pop on reset.)
    """
    feet_z = [forward_kinematics(sx, sy, *default_angles(sx, sy))["foot"][2]
              for _, sx, sy in LEGS]
    return float(-min(feet_z)) + FOOT_R


def summary() -> None:
    h = standing_height()
    print(f"trunk (l x w x h): {TRUNK['length']} x {TRUNK['width']} x {TRUNK['height']} m")
    print(f"segments: hip {HIP_LEN}  thigh {THIGH_LEN}  shank {SHANK_LEN}  foot_r {FOOT_R}")
    print(f"standing trunk height: {h:.3f} m")
    print(f"12 joints (order): {JOINTS}")
    print("\ndefault sprawl X-stance — foot positions in trunk frame (x fwd, y left, z up):")
    for name, sx, sy in LEGS:
        fk = forward_kinematics(sx, sy, *default_angles(sx, sy))
        foot = fk["foot"]
        knee = fk["knee"]
        bent = knee[2] > foot[2]  # knee above foot => leg is bent, not straight
        print(f"  {name}: foot=({foot[0]:+.3f}, {foot[1]:+.3f}, {foot[2]:+.3f})  "
              f"knee_z={knee[2]:+.3f}  bent={bent}")


if __name__ == "__main__":
    summary()
