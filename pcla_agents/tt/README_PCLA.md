# ThinkTwice → PCLA integration (`tt`)

[ThinkTwice](https://github.com/OpenDriveLab/ThinkTwice) — *"Think Twice before Driving:
Towards Scalable Decoders for End-to-End Autonomous Driving"* (CVPR 2023) — ported into PCLA
as the agent family **`tt`** (registry key `tt_tt`).

It is a closed-loop CARLA Leaderboard agent (4× RGB + LiDAR + IMU + GNSS + speedometer →
`carla.VehicleControl`). Its original ML stack (py3.7 / torch 1.12 / mmcv-full 1.7 /
mmdet 2.28 / mmdet3d 1.0 + compiled CUDA ops) is incompatible with PCLA's modern env, so the
model was **ported to PCLA's stack** — torch 2.2 / mmcv-lite 2.1 / mmengine 0.10, **without
mmdet, mmdet3d, or any nvcc-compiled op**.

## Status: ported & validated ✅

Validated in the **real PCLA conda env** (only `spconv` added):

- Full model **builds** from `configs/thinktwice.py` — 1344 param-tensors, **0 missing /
  0 unexpected** vs the released checkpoint.
- The released **515 MB checkpoint loads `strict=True`** (perfect key match).
- **Full forward pass runs** end-to-end on GPU: camera LSS + LiDAR (voxelization + sparse
  conv) + BEV fusion + deformable-attention decoder → `process_action` / `control_pid`
  produce steer/throttle/brake.
- The agent **instantiates exactly as PCLA loads it** (dynamic file load →
  `get_entry_point()` → build model → load checkpoint → build pipelines → 9 sensors).

Not yet exercised here: a **live closed-loop drive**, which needs a running CARLA server
(31-frame sensor queue + route planner). The model + agent are proven; the on-road behaviour
should be checked against the original once running in CARLA.

## Environment delta (the only change PCLA's env needs)

```bash
pip install spconv-cu120     # prebuilt wheel (ships CUDA, no nvcc); for the LiDAR sparse conv
```

Everything else the port needs is already in PCLA's env: torch 2.2, `mmcv-lite` 2.1,
`mmengine` 0.10, `torchvision` 0.17 (its `deform_conv2d` replaces the DCN op). **No mmdet3d,
no mmdet, no mmcv-full, no CUDA toolkit.**

## Usage

```python
agent = "tt_tt"                 # <agent>_<variant>
route = "./sample_route.xml"
pcla  = PCLA(agent, vehicle, route, client)
```

Checkpoint: download `thinktwice.pth` into `pcla_agents/tt_pretrained/` (see that folder's
README).

## How the port works

PCLA loads agents **in-process**, so the model must import under PCLA's env. The strategy
was to vendor the small set of OpenMMLab building blocks the model needs and replace every
compiled CUDA op with a wheel-shipped or pure-PyTorch equivalent — no code depends on
mmdet/mmdet3d/mmcv-ops at runtime.

| Concern | Original | Port |
|---|---|---|
| Registry / `build_*` | mmdet/mmdet3d registries | one local `mmengine` registry `TT_MODELS` (`code/tt_compat.py`) |
| `BaseModule`, fp16 decorators | `mmcv.runner` | `mmengine.model.BaseModule`; `force_fp32`/`auto_fp16` → no-ops (fp32 inference) |
| ResNet / PAFPN / SECOND / SECONDFPN / SparseEncoder / HardSimpleVFE | mmdet / mmdet3d | **vendored** under `code/vendored/` (source-faithful → checkpoint keys preserved) |
| `voxel_pooling` (BEVDepth CUDA op) | custom nvcc op | **pure-PyTorch** `index_add_` scatter (numerically exact), `ops/voxel_pooling/voxel_pooling.py` |
| Deformable attention (`mmcv.ops` CUDA) | `mmcv._ext` | **pure-PyTorch** reference `multi_scale_deformable_attn_pytorch` |
| DCN / deform conv (`mmcv.ops` CUDA) | `mmcv._ext` | **torchvision** `deform_conv2d` (`code/vendored/dcn.py`) |
| Sparse conv | spconv 1.x (mmcv) / spconv 2.x | **spconv 2.x** wheel (`spconv-cu120`); spconv layers registered like mmdet3d does |
| Voxelization (`mmcv.ops.Voxelization`) | custom CUDA op | **spconv `PointToVoxel`** (`code/model_code/backbones/lidarnet.py`) |
| `mmcv.parallel` DataContainer + collate | removed in mmcv 2.x | reimplemented in `code/tt_data.py` |
| Config / Compose / checkpoint | `mmcv.Config`, `mmdet3d` Compose, `mmcv.runner` | `mmengine.config.Config`, local `Compose`, `mmengine.runner.load_checkpoint` |
| deepspeed import crash | — | neutralized (its op probe shells to a missing nvcc); tt does no training |

The vendored modules under `code/vendored/` keep their original submodule/parameter names, so
the released checkpoint loads unchanged. `code/tt_compat.py` and `code/tt_data.py` are the
compatibility layers.

## Files changed for the integration

- `pcla_agents/tt/` — the ThinkTwice agent + ported model tree (`open_loop_training/`).
- `pcla_agents/tt_pretrained/` — checkpoint location (+ download helper).
- [`agents.json`](../../agents.json) — `"tt" → "tt"` entry.
- [`pcla_functions/give_path.py`](../../pcla_functions/give_path.py) — `tt` branch building the
  agent's `"<ckpt>+<config.py>"` conf string.
