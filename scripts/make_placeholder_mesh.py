#!/usr/bin/env python3
"""Generate a placeholder visual mesh for Phase 3 (no CAD tool needed).

Writes a rectangular box as a binary STL — a stand-in "tank deck" torso skin
that renders over the torso's collision sphere. This is throwaway: Phase 4
replaces it with CAD-authored tank parts. To reshape the torso silhouette you
edit HALF (here) and re-run; in Phase 4 you'd edit the CAD mesh instead.

Usage:
    ./mujoco-env/bin/python scripts/make_placeholder_mesh.py
"""

from __future__ import annotations

import os
import struct

import numpy as np

# Half-extents (x, y, z) in metres. Torso collision sphere is radius 0.25, so
# this comfortably encloses it. Wider/longer than tall => low, flat tank deck.
HALF = (0.32, 0.24, 0.12)
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "assets", "torso_shell.stl",
)


def box(half: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box centred at the origin: 8 verts, 12 triangles."""
    hx, hy, hz = half
    verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float64)
    faces = np.array([
        [1, 2, 6], [1, 6, 5],   # +x
        [0, 4, 7], [0, 7, 3],   # -x
        [3, 7, 6], [3, 6, 2],   # +y
        [0, 1, 5], [0, 5, 4],   # -y
        [4, 5, 6], [4, 6, 7],   # +z
        [0, 3, 2], [0, 2, 1],   # -z
    ], dtype=np.int64)
    return verts, faces


def write_binary_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write triangles with outward-facing normals (flip winding if needed)."""
    tris = verts[faces]                                   # (F, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    centroid = tris.mean(axis=1)                          # mesh is centred on origin
    flip = (n * centroid).sum(axis=1) < 0                 # normal points inward -> flip
    n[flip] *= -1
    tris[flip] = tris[flip][:, ::-1]                      # reverse vertex order
    with open(path, "wb") as f:
        f.write(b"\0" * 80)                               # 80-byte header
        f.write(struct.pack("<I", len(faces)))            # triangle count
        for i in range(len(faces)):
            f.write(struct.pack("<3f", *n[i]))
            for v in tris[i]:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))                 # attribute byte count


if __name__ == "__main__":
    v, faces = box(HALF)
    write_binary_stl(OUT, v, faces)
    full = tuple(round(2 * h, 3) for h in HALF)
    print(f"wrote {OUT}  (box {full} m, {len(v)} verts, {len(faces)} tris)")
