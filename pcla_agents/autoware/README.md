# Autoware as a PCLA agent

This makes **[Autoware](https://github.com/autowarefoundation/autoware)** — the open-source,
HD-map-based autonomous-driving stack — usable as a PCLA agent (`agent="autoware_v1"`), driving
a CARLA ego through PCLA just like the neural agents. Unlike them, Autoware is not an in-process
model: it is a full ROS 2 stack running in Docker, connected to CARLA by the
[evshary/autoware_carla_launch](https://github.com/evshary/autoware_carla_launch) Zenoh bridge
(**jazzy** branch: CARLA 0.9.16 / Autoware 1.8.0 / Zenoh 1.9.0).

Because Autoware is *map-first*, it needs a per-town HD map (a Lanelet2 lane map, a pointcloud
map for NDT localization, and a projector). This kit provides tooling to **generate those maps
from the running simulator**, so you can switch to any town — including custom/source-built ones —
without hunting for downloads.

## What works

| | Status |
|---|---|
| Drive a route (localize, plan, follow lanes, avoid obstacles) | ✅ any town |
| **Traffic lights** (stop at red, go on green) | ✅ **Town01**; other towns need a light-annotated map (see below) |
| **Map switching** (`TOWN=TownXX`) | ✅ any town with a lane map + a generated pointcloud |
| Localization | NDT, against a pointcloud map **generated from the sim** (no download) |

## How it works

- **PCLA owns the ego.** `sample_autoware.py` spawns the ego with `role_name="autoware_v1"`; the
  running Zenoh bridge attaches to it by that name and writes `VehicleControl` straight to CARLA.
- **The adapter** (`autoware_agent.py`) spawns the exact Autoware sensor kit onto the ego, then, on
  a background thread, initializes localization, sets the goal, and engages autonomous mode over
  `docker exec … ros2 …` (no ROS on the host). `run_step` just echoes `vehicle.get_control()`.
- **Traffic lights** are injected as ground truth: Autoware's camera classifier returns `UNKNOWN`
  on CARLA's rendered lights, so `carla_gt_bridge` reads each CARLA light's real state and publishes
  it to `/perception/traffic_light_recognition/traffic_signals` (mirroring a real V2I deployment).
  Autoware is launched with `use_traffic_light_recognition:=false` so the broken camera path stays off.
- **Coordinates:** Autoware's `map` frame is CARLA with the y-axis flipped (`y_aw = -y_carla`,
  `yaw_aw = -yaw_carla`). All the tooling uses this convention.

## One-time setup

Prereqs: an NVIDIA GPU + driver, Docker + nvidia-container-toolkit, a CARLA **0.9.16** server
(assumed at `~/carla_lite16/CarlaUE4.sh`), and the PCLA conda env with a CARLA **0.9.16** python
client. `kit/setup_autoware.sh` walks through it (clone the stack, build the two Docker images,
create the `aw_bridge`/`aw_autoware` containers, install this kit, fetch Town01's map):

```bash
bash pcla_agents/autoware/kit/setup_autoware.sh
```

The image build is large and GPU-bound; it defers to the upstream `make` targets and container
scripts. `kit/install_kit.sh` (called by setup, and re-runnable) deploys just the PCLA pieces
into an existing `autoware_carla_launch` clone.

## Running

Autoware needs its stack up first, so it is a two-step run (not the one-liner the neural agents use):

```bash
# 1) bring up exactly one CARLA + bridge + Autoware (headless is the most reliable):
bash ~/start_stack.sh
#    RVIZ=1 bash ~/start_stack.sh     # headless CARLA + Autoware RViz (car, map, lidar, path)

# 2) once it says READY, drive:
conda activate PCLA && cd ~/PCLA && python sample_autoware.py
```

**CARLA quality:** the stack renders at **Epic** by default (highest fidelity). On a lighter GPU
you can drop to Low quality:

```bash
CARLA_QUALITY=Low RENDER_OFFSCREEN=1 bash ~/start_stack.sh
```

(`CARLA_QUALITY` accepts CARLA's usual levels — `Low`, `Epic`. If the renderer segfaults or shows a
black window, add `CARLA_EXTRA_ARGS=-vulkan`.)

You can re-run `sample_autoware.py` repeatedly without restarting the stack (it resets the bridge
and re-initializes localization each run).

The per-run bring-up (docker access, fresh bridge, stale-ego cleanup) lives in
[`pcla_functions/autoware_runtime.py`](../../pcla_functions/autoware_runtime.py) —
`ensure_docker_access()`, `reset_bridge()`, `destroy_stale_ego()` (plus `has_docker_access()`).
Import them (`from pcla_functions import reset_bridge, destroy_stale_ego, ensure_docker_access`) if
you write your own driver loop instead of using `sample_autoware.py`.

## Routing & goals

**Autoware is goal-based, not route-based.** The neural PCLA agents are *fed* the dense waypoints
from `sample_route.xml` and follow them. Autoware instead takes a single **destination** and plans
its own lane-level route to it on the HD map — it does *not* trace those waypoints. So "giving it a
route" means "giving it a destination" (plus, optionally, via-points to shape the path).

The destination is chosen in [config.yaml](config.yaml), in this priority order:

| Setting | Meaning |
|---|---|
| `goal_override: [x, y, yaw_deg]` | Explicit destination. Highest priority. |
| `goal_ahead_m: N` | Drive N metres forward from the spawn (no coordinates needed). |
| *(neither set)* | The **endpoint** of `sample_route.xml` (its last point). |

All poses are in Autoware's `map` frame — **CARLA with the y-axis flipped**:
`(x_aw, y_aw, yaw_aw) = (x_carla, -y_carla, -yaw_carla)`. To get coordinates, read a CARLA
`Transform` (a spawn point, or move the spectator there) and flip the sign of `y`/`yaw`; or, with
`RVIZ=1`, click **2D Goal Pose** in RViz.

### Checkpoints (multi-leg routes)

Because Autoware picks its own lanes to the goal, to steer it down *particular* streets you add
ordered **via-points**. List them in `config.yaml` (Autoware map frame, same `[x, y, yaw_deg]`):

```yaml
checkpoints:
  - [120.0, -55.0, 90.0]
  - [90.0, -110.0, 180.0]
```

The adapter publishes them to `/planning/mission_planning/checkpoint` after the goal, so the route
becomes `ego → cp1 → cp2 → … → goal` (Autoware still plans lane-by-lane between them). Order
matters; checkpoints work with any of the goal sources above.

## Switching towns

```bash
# Prepare a town once (downloads its lane map, generates its pointcloud + traffic-light table):
bash ~/PCLA/pcla_agents/autoware/kit/prepare_town.sh Town03
# Then run the stack on it:
TOWN=Town03 bash ~/start_stack.sh
TOWN=Town03 python sample_autoware.py   # TOWN only affects start_stack; the sample reads whatever is loaded
```

`prepare_town.sh` needs CARLA running (it stops the bridge, loads the town, and sweeps a LiDAR to
build the pointcloud map). Assets land in `carla_map/<Town>/`.

### Traffic lights on other towns

Traffic lights work on **Town01** because its shipped map (evshary's) annotates them. The community
lane maps used for other towns (Town02–07, Town10HD) are **lane-only** — they have no traffic-light
regulatory elements, so Autoware has nothing to obey there and the kit disables light injection for
those towns automatically. To enable lights on another town you must add `traffic_light` regulatory
elements to its `lanelet2_map.osm` (e.g. in TIER IV's Vector Map Builder, or a future auto-annotation
tool that derives them from CARLA's stop-line data). Custom/source-built towns are the same: driving
works from the generated pointcloud; lights need annotations in the map.

### Custom / source-built towns

There is no download for a custom town. Export its road network with
`world.get_map().to_opendrive()`, convert OpenDRIVE→Lanelet2 (e.g. CommonRoad's `opendrive2lanelet`),
drop the `.osm` in `carla_map/<Town>/`, then run `prepare_town.sh <Town>` to generate the pointcloud
and (empty, unless you annotate lights) traffic-light table.

## Kit contents (`kit/`)

| File | Role |
|---|---|
| `carla_gt_bridge.py` | Read-only node: mirrors CARLA's real light states into Autoware's `traffic_signals` topic. Installed into `aw_autoware`. |
| `gen_pointcloud.py` | Builds a town's NDT pointcloud map by sweeping a LiDAR over its roads. |
| `gen_map_info.py` | Builds the CARLA-light → Autoware-group table from a town's Lanelet2 map. |
| `prepare_town.sh` | Orchestrates a town's map prep (osm + projector + pointcloud + table). |
| `start_stack.sh` | One-shot clean bring-up of CARLA + bridge + Autoware (verifies a single instance). |
| `run_autoware_headless.sh` / `run_autoware_rviz.sh` | Autoware launch wrappers (camera TL off). |
| `install_kit.sh` | Deploys the runtime pieces into an `autoware_carla_launch` clone. |
| `setup_autoware.sh` | One-time end-to-end setup. |

## Troubleshooting

- **Car doesn't move / "routing never accepted":** ensure your shell has Docker access — the adapter
  drives Autoware via `docker exec`. `sample_autoware.py` re-execs under `sg docker` if needed;
  otherwise log out/in once to join the `docker` group.
- **Only run one stack:** two Autoware launches double `/clock` and wedge discovery. `start_stack.sh`
  restarts the containers to guarantee a single instance. Don't hand-poke a running stack.
- **Windowed CARLA is jittery** (can trip Autoware's freshness monitors → braking). Use
  `RENDER_OFFSCREEN=1` (headless) — with `RVIZ=1` for a visual.
- **`load_world` times out** after stopping the bridge: CARLA was left frozen in sync mode; the tools
  unfreeze it (async + tick) before loading. If CARLA segfaults after many restarts, just re-run
  `start_stack.sh`.
