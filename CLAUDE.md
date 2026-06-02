# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A custom MuJoCo quadruped ("**spyder**") built up from a **byte-identical copy of Gymnasium's `Ant-v5`**. The progression is: (1) reproduce stock Ant exactly, (2) view it, (3) make small parametric edits toward a cooler, **3D-printable**, mesh-based 4-legged robot whose silhouette/stance evokes `docs/media/4_leg_robot.png` (a heavy, tank-like quadruped).

Two consumers drive every design decision:
- **RL training in Gymnasium/MuJoCo (CPU)** — must stay drop-in compatible with `Ant-v5`.
- **GPU training (Isaac Lab + MuJoCo Playground/MJX)** and physical 3D printing — favor **URDF + mesh files** as the portable source-of-truth, since URDF imports into MuJoCo, Isaac, PyBullet, Gazebo, and Drake.

## Environment & commands

One venv: `mujoco-env` (Python 3.14.4) pinning `gymnasium==1.3.0`, `mujoco==3.9.0`, `glfw==2.10.0`, `numpy==2.4.6` (+ `imageio`). There is no system `python`/`gymnasium` — bare `python3` fails with `ModuleNotFoundError`. Always prefix with the venv path.

macOS viewer gotcha: MuJoCo's **managed** `mujoco.viewer.launch()` crashes with `RuntimeError: Caught an unknown exception` in `_Simulate` (independent of Python version). The fix is `launch_passive()` + a self-driven step loop, which `scripts/view.py` uses. Any live window must run under **`mjpython`** (owns the Cocoa main thread); headless rendering runs under plain `python`.

```bash
# Prove a custom XML is still drop-in Ant-v5 (obs (105,), action (8,), skeleton inventory):
./mujoco-env/bin/python scripts/check_compat.py [models/spyder.xml]

# Live interactive viewer (orbit/poke); --physics steps torques so it moves:
./mujoco-env/bin/mjpython scripts/view.py
./mujoco-env/bin/mjpython scripts/view.py --physics --policy gentle

# Headless — render a rollout to GIF/MP4, no window (.mp4 needs `pip install imageio-ffmpeg`):
# (the committed preview lives at docs/media/spyder_preview.gif; other *.gif are ignored)
./mujoco-env/bin/python scripts/view.py --save docs/media/spyder_preview.gif --policy gentle
```

Stock reference model (read, never edit): `mujoco-env/lib/python3.14/site-packages/gymnasium/envs/mujoco/assets/ant.xml`
Env wrapper: `…/gymnasium/envs/mujoco/ant_v5.py`

## The v5-compatibility contract (do not break)

`AntEnv` computes its observation space from the model at load time
(`obs = qpos + qvel + cfrc_ext[1:]`). To stay **identical to Ant-v5** a custom XML must keep:

- **1 free joint** (`root`) on the torso + **8 hinge joints** (named `hip_1..4`, `ankle_1..4`).
- **13 bodies** total → observation `Box(105,)`, action `Box(8,)`.
- **8 `motor` actuators**, `ctrlrange="-1 1"`, in the **same order** the stock file lists them (hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3) — action-index meaning is positional.

You may freely change geom *sizes/shapes/materials/positions* and joint *ranges*. You may NOT add/remove joints or bodies without versioning it as a new env (e.g. `Spyder-v0`) rather than calling it v5.

Pass-through kwargs that matter when geometry changes:
- `main_body` (default `1`, the torso body id) — keep the torso as body 1, or pass `main_body="torso"`.
- `healthy_z_range=(0.2, 1.0)` — torso z must stay in range or the episode terminates instantly. Stock torso sits at `pos z=0.75` with `init_qpos` z=`0.55`. **Resize the body ⇒ re-tune these.**

## Ant MJCF cheat-sheet (what a custom file inherits)

`compiler angle="degree" coordinate="local" inertiafromgeom="true"`; `option integrator="RK4" timestep="0.01"` (env `frame_skip=5` ⇒ `dt=0.05`). Defaults: `joint armature=1 damping=1 limited=true`; `geom condim=3 density=5 friction="1 0.5 0.5"`. Torso = sphere `size=0.25`; each leg = three `capsule` geoms (`size=0.08`) across an aux body (hip hinge, range `-30 30`, axis `0 0 1`) and a lower link (ankle hinge); all 8 motors `gear=150`.

## Target architecture (menagerie-style)

Follow the `mujoco_menagerie` layout once meshes enter the picture:
- `scene.xml` — includes the robot + floor + skybox/haze/lighting (the thing you view/train).
- `<robot>.xml` — the model only; `assets/` holds STL/OBJ meshes; extract shared props into `<default>` classes.
- **Separate visual geoms from collision geoms** (visual = pretty/print mesh; collision = simple capsules/boxes the physics uses). This keeps RL fast and gives clean STLs for printing.
- Optional `<robot>_mjx.xml` with sphere-only collisions for MJX/GPU.
- Menagerie quadrupeds (Go2, ANYmal, Spot, Barkour) are **12-DoF (3 joints/leg)** — use them as a *styling/asset* reference only; they are NOT drop-in for the 8-DoF Ant skeleton.

## Key reference docs (point at the specific page for the task)

Docs pages are large — fetch the one relevant to the current task, don't ingest everything.

Designing the robot (MJCF authoring — **this is where we start**):
- MJCF XML reference (the spec for valid robot XML): https://mujoco.readthedocs.io/en/stable/XMLreference.html
- Modeling guide (concepts behind the reference): https://mujoco.readthedocs.io/en/stable/modeling.html
- MuJoCo Menagerie (highest-leverage: clone & adapt a real model rather than generating MJCF from scratch): https://github.com/google-deepmind/mujoco_menagerie

Training (later phases):
- MuJoCo Playground repo (live code/envs/training scripts — more useful than the bundled PDF): https://github.com/google-deepmind/mujoco_playground
- MJX→Warp migration (`impl='warp'` flag, newer than the report): https://github.com/google-deepmind/mujoco_playground/discussions/197
- MJX docs (JAX/Warp backend; note the Mac-vs-NVIDIA backend split): https://mujoco.readthedocs.io/en/stable/mjx.html

## Repo contents

- `models/` — the robot MJCF (`spyder.xml`) and `assets/` for future STL/OBJ meshes.
- `scripts/` — `check_compat.py` (v5 contract) and `view.py` (viewer + recorder).
- `docs/references/` — external papers/reports (e.g. the MuJoCo Playground MJX/GPU-RL report).
- `docs/media/` — images/clips for the README & docs: `4_leg_robot.png` (visual target), `spyder_preview.gif` (preview render).
- `mujoco-env/` — the virtualenv (gitignored; do not commit/edit its site-packages).
- `TODO_spyder.md` — phased plan (local-only, gitignored).
