# Spyder-v0 design — the 12-DoF spider (`models/spyder12.xml`)

Why the robot looks the way it does, what its contract is, and which numbers to
touch when tuning. Written to be readable without a robotics background.

## 1. Where it comes from

Gymnasium's `Ant-v5` is already a 4-legged spider in silhouette — sphere torso,
four capsule legs in an X — but each leg has only **2 joints**:

| Ant joint | axis | what it does |
|---|---|---|
| `hip_i` | vertical (yaw) | sweeps the whole leg forward/back |
| `ankle_i` | horizontal | folds the outer segment |

A foot's position in space is **three numbers** (forward/back, up/down,
near/far), and each joint controls exactly one number. With 2 joints the foot
is confined to a thin 1-D curve of reachable points: the Ant can shuffle and
turn, but it cannot lift a foot to *place* it — no stepping onto rocks, no
crouching, no real terrain.

`spyder12.xml` keeps the Ant's body, X-stance, capsule construction, and
physics defaults, and adds **one lift joint per leg** — the same fix real
arthropods use (coxa–femur–tibia). 4 legs × 3 joints = **12 DoF**:

| Spyder joint | axis | marker | what it does |
|---|---|---|---|
| `hip_i` | vertical (yaw), `0 0 1` | 🔴 red | sweeps leg forward/back (unchanged from Ant) |
| `lift_i` | horizontal pitch | 🟢 green | raises/lowers the whole leg — **the new joint** |
| `knee_i` | horizontal pitch | 🔵 blue | folds the lower leg (the Ant's ankle, re-homed) |

Three joints per leg is the *minimum* that lets a foot reach any point in its
workspace — which is why every serious quadruped (Go2, ANYmal, Spot) uses
exactly 3 per leg. More joints (a real spider has ~7/leg) only add redundant
ways to reach the same spots, at the cost of motors, weight, and a harder RL
problem.

The colored marker spheres in the model are massless, collision-free visuals —
they exist purely so renders document the joint layout (red/green/blue =
hip/lift/knee, matching the tables above).

## 2. Leg geometry (per leg, local units)

Legs radiate diagonally (`±0.2, ±0.2` from torso center), numbered like Ant:
1 = front-left, 2 = front-right, 3 = back-left, 4 = back-right.

```
torso (sphere r=0.25, z=0.35)
 └─ coxa stub   0.20 out          (welded — part of the torso body)
    └─ [hip]    coxa   0.12 out            ← yaw,  range ±40°
       └─ [lift] femur 0.22 out, 0.25 UP   ← pitch, range ±60°
          └─ [knee] tibia 0.22 out, 0.52 DOWN ← pitch, range ±50°
```

The femur angles *up* and the tibia *down* — the high-knee arch that makes it
read as a spider and gives the swing leg ground clearance. At the authored
pose (all joints 0°) each foot capsule is exactly tangent to the floor:
torso 0.35 + femur 0.25 − tibia 0.52 = 0.08 = capsule radius.

Body count stays **13** (torso + 3 bodies/leg), same as Ant, by welding the
coxa stubs into the torso — so AntEnv's contact-force observation block keeps
the familiar 13×6 layout.

## 3. Physics choices (and why)

* **Ant defaults kept:** `integrator RK4, timestep 0.01`, joint
  `armature 1 damping 1`, geom `density 5, friction 1 0.5 0.5`, motors
  `gear 150, ctrlrange ±1`. Total mass ≈ 1.09 kg (Ant ≈ 0.91).
* **One deliberate addition — joint springs (`stiffness 30`):** springs pull
  every hinge back to the authored stance. Without them, an arched leg buckles
  under gravity at zero torque (the flat Ant doesn't have this problem —
  splayed legs are passively stable). With them the robot stands indefinitely
  at zero control (verified: 1000/1000 steps, ~2.5° sag, torso settles
  z ≈ 0.334), giving RL a sane starting posture and making torque commands act
  as offsets around the stance.

## 4. The Spyder-v0 contract (`envs/spyder_env.py`)

Registered on Gymnasium's generic `AntEnv` — same reward (forward velocity −
ctrl/contact costs), same healthy-z termination; only the model and kwargs
change. **Not** drop-in `Ant-v5` (that contract lives in `models/spyder.xml`
and is checked by `scripts/check_compat.py`).

| item | value |
|---|---|
| action space | `Box(12,)` — order `hip_1, lift_1, knee_1, hip_2, … knee_4` (positional!) |
| observation | `Box(113,)` = qpos(17) + qvel(18) + `cfrc_ext[1:]`(13×6=78) |
| skeleton | 1 free root + 12 hinges, 13 bodies, 12 motors |
| `main_body` | `"torso"` |
| `healthy_z_range` | `(0.15, 1.0)` — terminate on belly-flop, never on jumping |
| `frame_skip` | 5 → dt = 0.05 s, same cadence as Ant |
| `reset_noise_scale` | 0.05 (gentler than Ant's 0.1 — the arched pose tolerates less) |

Guarded by `scripts/check_spyder.py`: skeleton inventory, env shapes, and a
zero-torque stand test (mean episode length ≥ 200 of 1000).

## 5. Tuning knobs

| symptom | knob |
|---|---|
| episodes end instantly after a geometry edit | re-tune `HEALTHY_Z` in `envs/spyder_env.py` (torso rest ≈ 0.33) |
| legs buckle at rest / stance too soft | joint `stiffness` in the `<default>` block (30) |
| motions too violent / too weak in RL | motor `gear` (150) |
| feet slip | geom `friction` (first term, 1.0) |
| stance wider/narrower | the `±0.2` leg-attach offsets and segment `fromto`s (keep foot-tangency: torso_z + femur_rise − tibia_drop = 0.08) |

## 6. What this unlocks (and what it doesn't yet)

The lift joint gives the three primitives terrain work is built from — step
*height*, foot *placement*, body-*height* control. Running, mud, rock fields,
and backflips are those primitives done fast and in rhythm by a trained
policy. But a policy only learns what the sim contains: flat-floor training
(`scene12.xml`) teaches flat-floor running. Terrain skills need terrain in the
scene — heightfields, boxes/ramps, randomized friction — which is the next
phase (MuJoCo Playground / Isaac-style domain randomization).

Printable meshes are a later, purely-visual layer: menagerie convention keeps
collision capsules for physics and adds STL visual geoms on top, so RL speed
and the printed look stay decoupled.
