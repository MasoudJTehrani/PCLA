# PCLA Bridge: run PCLA agents without CARLA

PCLA agents were written for the CARLA Leaderboard, but their real per-frame
contract is environment-agnostic:

This bridge lets you drive a PCLA agent from *any* source of
sensor data and route it to any sink:

- a **real vehicle** (sensors → drive-by-wire),
- **another simulator** — BeamNG, LGSVL/AWSIM, Gazebo, a custom rig,
- a **recorded log / video**, for offline evaluation.

---

## Files

| file | what it does |
|---|---|
| [`pcla_bridge.py`](pcla_bridge.py) | The bridge: `PCLAAgentBridge` — load an agent, feed sensors, get control |
| [`bridge_sample.py`](bridge_sample.py) | Sample code of running an agent over a recorded drive |
| [`samples/carla_run/`](samples/) | A bundled real Town02 recording (video + route + telemetry) to try immediately |

---

## Using the bridge in your own code

Acticate the PCLA environment first.

```python
from pcla_bridge import PCLAAgentBridge

bridge = PCLAAgentBridge("neat_aim2dsem")          # any sensor-only agent
print(bridge.sensor_layout())                       # give you exactly what sensors to supply

bridge.set_route([(52.5200, 13.4040), (52.5205, 13.4040), ...])   # GPS waypoints

while driving:
    control = bridge.step(sensor_readings, dt=0.05)  # sensor_readings: {id: value} and delta-time 20fps
    apply(control.throttle, control.steer, control.brake)
```

`sensor_readings` is a dict keyed by the ids from `sensor_layout()`, each in its natural form: 
- camera → HxWx3 **BGR**; 
- gnss → `[lat,lon,alt]`; 
- imu → `[ax,ay,az,gx,gy,gz,compass]`; 
- speed → m/s

The bridge converts to CARLA's exact wire format for you.

## Sample code

A real recorded sample (`samples/carla_run/`) ships with the repo, so you can run the
bridge immediately without a CARLA server. From this folder, with the `PCLA` conda
env active:

```bash
python bridge_sample.py --agent neat_aim2dsem --sample samples/carla_run --overlay
```

You'll get `samples/carla_run/replay_out/control.csv` and `overlay.mp4`. The overlay
draws the throttle/steer/brake the agent produced on each frame, over the actual
camera footage, so you can *watch* it react. `neat_aim2dsem` is a good first target:
a single front camera (+ IMU / GPS / speed), **no LiDAR**.

---

## Applying it to your own application

The point of the bridge is to drive an agent from **your** data, your simulator,
your vehicle, your logs. Export a sample directory in the layout below and point
`bridge_sample.py` at it, or call the bridge directly in your own loop (see the
snippet under "Using the bridge in your own code" above). A sample is just:

```
<sample>/
  route.json       # {"waypoints": [[lat,lon], ...]}   the global plan
  video.mp4        # front-camera frames, one per telemetry row
  telemetry.jsonl  # per-frame {gps:[lat,lon,alt], speed, imu:[7], dt}
```

The bundled `samples/carla_run/` is one such export, recorded from a CARLA run as a
worked example. One thing to get right when you make your own: camera agents steer
toward a **GPS-derived target point**, so the camera, GPS and route must be
*consistent frame-for-frame*. A video with an unrelated GPS route will load fine but
won't steer sensibly.

---


### Which agents work

**Runnable:** the sensor-only agents:
`tt`, `minddrive`, `orion`, `simlingo`, `lmdrive`, `interfuser`, `neat*`, `lav*`, `tfv3/4/5/6`, `wor`, `lbc`. 
The bridge reads each agent's own `sensors()` and adapts automatically; it does **not** fabricate a
missing sensor (you get a clear error). Names are exactly as in `agents.json` / `sample.py`, including seed suffixes (`tfv4_lav_0`).

**Blocked** (with a clear message): 
privileged agents (`plant`, `plant2`, `carl`, `roach`) read ground-truth pose + a simulator BEV map; and `autoware`, which is an external ROS 2 stack, not a `run_step` model.

---

## The three things the bridge can't do for you

1. **Sensor fidelity.** Match resolution, FOV, placement and rate to what
   `sensor_layout()` declares. A mismatch loads fine and steers badly.
2. **Localization / frames.** Agents match live GPS against your route in a metric
   frame. In a simulator you control both ends; on a real vehicle this is a genuine
   localization task — validate, vehicle stationary at a known point, that the
   agent's computed position lands where you expect before trusting any steering.
3. **Actuation + safety.** Mapping normalized throttle/steer/brake to your
   vehicle/sim, behind an e-stop, is yours.
