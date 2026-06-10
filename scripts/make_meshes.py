#!/usr/bin/env python3
"""Author the spyder tank meshes in build123d (OpenCASCADE solids) -> STL.

One STL per rigid link, all dimensioned from `spyder_params`. Each part is
modelled in the local frame its MuJoCo body expects (see spyder_params docstring),
so the STL drops straight in.

The visual language is the LEO tank-mech reference (docs/media/4_leg_robot.png):
faceted boxes with hard chamfers, inset panel lines, a two-tier hull (lower hull
with a 45° glacis + raised turret deck), and legs built around a huge flat thigh
armor plate. Build order matters with OpenCASCADE: chamfer the plain solid FIRST,
subtract pockets LAST — chamfering edges that cross pocket boundaries fails.

build123d needs Python <3.14, so this runs in the dedicated `cad-env` (3.13),
NOT the 3.14 training venv. The STLs it writes are static artifacts the trainer
and renderer (mujoco-env) consume:

    ./cad-env/bin/python scripts/make_meshes.py        # (re)generate meshes
    ./mujoco-env/bin/python scripts/render_robot.py    # then preview them
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find spyder_params

from build123d import (  # noqa: E402
    Axis, Box, BuildPart, Cylinder, Locations, Mode, Pos, Rotation, Sphere,
    chamfer, export_stl,
)

import spyder_params as P  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "assets")


def _build_turret():
    """Raised upper deck: a smaller faceted box with its own glacis + sensor block."""
    Tl, Tw, Th = P.TURRET["length"], P.TURRET["width"], P.TURRET["height"]
    with BuildPart() as p:
        Box(Tl, Tw, Th)
        # turret glacis: hard chamfer on the front-top edge
        front_top = p.edges().filter_by(Axis.Y).group_by(Axis.X)[-1].sort_by(Axis.Z)[-1]
        chamfer(front_top, length=0.035)
        # softer bevels on the remaining top edges (sides + rear)
        side_top = p.edges().filter_by(Axis.X).group_by(Axis.Z)[-1]
        chamfer(side_top, length=0.022)
        rear_top = p.edges().filter_by(Axis.Y).group_by(Axis.X)[0].sort_by(Axis.Z)[-1]
        chamfer(rear_top, length=0.022)
        # sensor block riding on the turret roof, slightly forward
        with Locations(Pos(0.04, 0, Th / 2 + 0.045 / 2)):
            Box(0.07, 0.10, 0.045)
    return p.part


def build_trunk():
    """Two-tier tank hull: glacis + side skirts below, faceted turret deck above."""
    L, W, H = P.TRUNK["length"], P.TRUNK["width"], P.TRUNK["height"]
    with BuildPart() as p:
        Box(L, W, H)
        # sloped front glacis: 45° chamfer on the front-top edge (||Y, max X, max Z)
        front_top = p.edges().filter_by(Axis.Y).group_by(Axis.X)[-1].sort_by(Axis.Z)[-1]
        chamfer(front_top, length=0.10)
        # gentler rear-top bevel
        rear_top = p.edges().filter_by(Axis.Y).group_by(Axis.X)[0].sort_by(Axis.Z)[-1]
        chamfer(rear_top, length=0.05)
        # sloped side-skirt shoulders (top edges ||X)
        side_top = p.edges().filter_by(Axis.X).group_by(Axis.Z)[-1]
        chamfer(side_top, length=0.04)
        # hip mounting bosses: axle-hub cylinders along x at the 4 lower corners
        for sx in (1, -1):
            for sy in (1, -1):
                with Locations(Pos(P.HIP_X * sx, P.HIP_Y * sy, P.HIP_Z) * Rotation(0, 90, 0)):
                    Cylinder(radius=0.055, height=0.16)
        # pockets LAST (panel lines): inset side-skirt panels
        for sy in (1, -1):
            with Locations(Pos(0, sy * W / 2, -0.015)):
                Box(L * 0.5, 0.025, H * 0.45, mode=Mode.SUBTRACT)
        # visor slot recessed into the glacis face (the amber strip sits proud in it)
        with Locations(Pos(L / 2 - 0.05, 0, 0.02) * Rotation(0, 45, 0)):
            Box(0.012, 0.34, 0.02, mode=Mode.SUBTRACT)
    # turret deck rides on top, set back from the glacis
    turret = Pos(-0.05, 0, H / 2 + P.TURRET["height"] / 2 - 0.005) * _build_turret()
    return p.part + turret


def build_hip():
    """HAA shoulder pod: an octagonal armored barrel along y with an axle hub."""
    with BuildPart() as p:
        Box(0.10, P.HIP_LEN * 2, 0.10)
        # chamfer the 4 long edges -> octagonal cross-section
        chamfer(p.edges().filter_by(Axis.Y), length=0.022)
        # joint hub greeble on the HAA (x) axis; pokes through as a ring
        with Locations(Rotation(0, 90, 0)):
            Cylinder(radius=0.052, height=0.04)
    return p.part


def build_thigh():
    """Upper leg: the signature flat armor plate, nearly hull-height.

    Hangs along -z; the plate rises `top_ext` ABOVE the HFE origin so it covers
    the hip pod, which is what sells the "knees as tall as the hull" read.
    Kept y-symmetric so one STL serves both left and right legs.
    """
    w, t, ext = P.THIGH_PLATE["w"], P.THIGH_PLATE["t"], P.THIGH_PLATE["top_ext"]
    plate_h = P.THIGH_LEN + ext
    with BuildPart() as p:
        with Locations(Pos(0, 0, ext - plate_h / 2)):
            Box(w, t, plate_h)
        # facet the slab. Order matters: bevel the clean top/bottom loops of the
        # plain box FIRST, then the verticals — chamfering a loop whose corners
        # already end on chamfer faces makes OpenCASCADE give up.
        chamfer(p.edges().group_by(Axis.Z)[-1], length=0.012)
        chamfer(p.edges().group_by(Axis.Z)[0], length=0.012)
        chamfer(p.edges().filter_by(Axis.Z), length=0.022)
        # knee knuckle bridging to the shank (pokes through both faces)
        with Locations(Pos(0, 0, -P.THIGH_LEN) * Rotation(90, 0, 0)):
            Cylinder(radius=0.034, height=0.075)
        # inset armor panels on BOTH faces (pockets last; keeps the mesh y-symmetric)
        for sy in (1, -1):
            with Locations(Pos(0, sy * t / 2, -0.06)):
                Box(0.11, 0.012, 0.15, mode=Mode.SUBTRACT)
    return p.part


def build_shank():
    """Lower leg: thick faceted block down -z with a broad foot pad.

    The visible foot tip is a sphere at -SHANK_LEN matching the collision sphere:
    there is no ankle joint and the shank stands tilted, so a flat collision foot
    would land on a corner edge — the sphere is the stable contact, the pad is
    armor styling above it.
    """
    w, t = P.SHANK_BLOCK["w"], P.SHANK_BLOCK["t"]
    fp = P.FOOT_PAD
    with BuildPart() as p:
        with Locations(Pos(0, 0, -P.SHANK_LEN / 2)):
            Box(w, t, P.SHANK_LEN)
        chamfer(p.edges().filter_by(Axis.Z), length=0.02)
        # broad foot pad (wedge): chamfered bottom loop
        with Locations(Pos(0.01, 0, -P.SHANK_LEN + fp["h"] / 2 - 0.01)):
            Box(fp["x"], fp["y"], fp["h"])
        chamfer(p.edges().group_by(Axis.Z)[0], length=0.012)
        # foot tip — mirrors the collision sphere so the visual touches the ground
        with Locations(Pos(0, 0, -P.SHANK_LEN)):
            Sphere(P.FOOT_R * 0.95)
    return p.part


def build_sensor():
    """Turret optic dome (its own mesh so it can take the accent colour)."""
    with BuildPart() as p:
        Sphere(0.03)
    return p.part


def main():
    os.makedirs(ASSETS, exist_ok=True)
    parts = {
        "trunk": build_trunk(),
        "hip": build_hip(),
        "thigh": build_thigh(),
        "shank": build_shank(),
        "sensor": build_sensor(),
    }
    for name, part in parts.items():
        path = os.path.join(ASSETS, f"{name}.stl")
        export_stl(part, path, tolerance=0.0008, angular_tolerance=0.1)
        sz = part.bounding_box().size
        print(f"{name:7s} -> {os.path.relpath(path):28s}  bbox {sz.X:.3f} x {sz.Y:.3f} x {sz.Z:.3f} m")


if __name__ == "__main__":
    main()
