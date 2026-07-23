"""
pcla_bridge.py — drive a PCLA agent from ANY sensor source (no CARLA server).

PCLA agents were written for the CARLA Leaderboard, but their actual per-frame
contract is environment-agnostic:

    control = agent.run_step(input_data, timestamp, vehicle=None)

where
    input_data : { sensor_id: (frame_int, numpy_array) }   # CARLA's data layout
    control    : carla.VehicleControl(throttle in [0,1], steer in [-1,1], brake in [0,1])

`carla` here is only a pip package used for its data *types* — no server, no world,
no ticking. So this bridge lets you run a PCLA agent against whatever can produce
those sensor readings and consume that control:

    * a real vehicle (cameras/LiDAR/GNSS/IMU -> drive-by-wire actuators);
    * another driving simulator — BeamNG, LGSVL/AWSIM, Gazebo, a custom rig;
    * a recorded log / video for offline evaluation.

The bridge is the source-independent middle: it loads one agent standalone, turns
your sensor readings into the exact dict the agent expects, and returns the control
it wants to apply. Capturing sensors and consuming the control on your side is the
"integration" — it lives OUTSIDE this file and is specific to your source/sink.

────────────────────────────────────────────────────────────────────────────────
⚠️  SAFETY (real vehicles). These are research models trained only in simulation,
    with no safety guarantees and a large sim-to-real gap. If your sink is a real
    car, only ever run this on a closed track or a bench rig, with a hardware
    e-stop and a human able to take over instantly — never on public roads. This
    bridge is deliberately *just* the software plumbing; the safety envelope around
    it is your responsibility. (For a purely virtual sink this is moot, but the
    sim-to-sim domain gap still applies to how the agent behaves.)
────────────────────────────────────────────────────────────────────────────────

What this file gives you:
  * standalone loading of a sensor-only PCLA agent (reuses PCLA's own agent
    resolution + per-agent module isolation, so no CARLA server is needed);
  * a clock driver (agents ask for a monotonically increasing timestamp);
  * a route setter that builds the global plan agents expect;
  * a sensor marshaller: your "natural" readings -> CARLA's exact input_data dict.

What YOU must still provide (marked `# >>> INTEGRATION`):
  * sensor capture from your source, at the resolutions/rates the agent's
    sensors() declares;
  * a route (list of GPS waypoints) for where you want to go;
  * alignment of your source's positioning to the route's frame (a localization
    problem — see the note in set_route);
  * applying the returned control in your source (actuators, or the sim's
    vehicle-control input).

Privileged agents (plant, plant2, carl, roach) are BLOCKED: they read ground-truth
ego pose and a bird's-eye-view map from the CARLA simulator specifically, which a
generic sensor source does not provide.
"""

import importlib.util
import math
import os
import sys

import numpy as np

# carla is imported for its data *types* (VehicleControl, Transform, ...).
# It does NOT need a running server.
import carla


# Agents that read simulator ground-truth (exact pose and/or BEV semantic maps).
# They cannot be driven from a generic sensor source and are refused up front.
PRIVILEGED_AGENTS = {"plant", "plant2", "carl"}

# Agents that are not sensor-in / control-out models at all, so this bridge does
# not apply. `autoware` is a thin adapter to an external ROS 2 Autoware stack
# running in Docker (it plans/controls itself); driving it means running that
# stack, not calling run_step().
UNSUPPORTED_AGENTS = {"autoware"}

# Sensor ids the agent may declare but that need not be supplied every step:
# pseudo/map sensors and event sensors (collision fires only on impact, so an
# empty stream is normal). Marshalling tolerates these being absent.
OPTIONAL_SENSOR_TYPES = ("sensor.opendrive_map", "sensor.collision")


def _find_pcla_root(start=None):
    """Locate the PCLA repo root (the dir containing agents.json) by walking up.

    This file lives in PCLA/pcla_bridge/, so the root is a parent directory. We
    search rather than hard-code the depth, so moving this file again doesn't
    silently break the default.
    """
    d = start or os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "agents.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # Fallback: assume the conventional PCLA/pcla_bridge/ layout.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _StepTimestamp:
    """Minimal stand-in for carla's timestamp, enough to drive GameTime."""
    def __init__(self, frame, elapsed_seconds, delta_seconds):
        self.frame = frame
        self.elapsed_seconds = elapsed_seconds
        self.delta_seconds = delta_seconds


class PCLAAgentBridge:
    """Load one PCLA agent and drive it frame-by-frame from any sensor source.

    Args:
        agent_name : "<agent>_<variant>" exactly as in agents.json / sample.py,
                     e.g. "tfv4_lav", "interfuser_if", "minddrive_05b".
        pcla_dir   : PCLA repo root. Defaults to this file's directory.
    """

    def __init__(self, agent_name, pcla_dir=None):
        self.pcla_dir = pcla_dir or _find_pcla_root()
        if self.pcla_dir not in sys.path:
            sys.path.insert(0, self.pcla_dir)

        base = agent_name.split("_")[0]
        if base in UNSUPPORTED_AGENTS:
            raise ValueError(
                f"Agent family '{base}' is not a sensor-in / control-out model — it is "
                f"an adapter to an external self-driving stack (e.g. Autoware over ROS 2 "
                f"in Docker) that plans and controls itself. There is no run_step() to "
                f"feed sensor frames into, so this bridge does not apply.")
        if base in PRIVILEGED_AGENTS:
            raise ValueError(
                f"Agent family '{base}' is privileged: it reads ground-truth ego "
                f"pose and/or a simulator BEV map that a generic sensor source does "
                f"not provide. It cannot be driven through this bridge. Use a sensor-only "
                f"agent (transfuser*, interfuser, neat, lav, wor/lbc, tt, "
                f"minddrive, orion, simlingo, lmdrive).")

        from pcla_functions import give_path, clear_agent_modules
        from leaderboard_codes.timer import GameTime

        self._GameTime = GameTime
        GameTime.restart()

        # Resolve agent + config paths and load the agent module in isolation,
        # exactly like PCLA.setup_agent does (so vendored forks don't collide).
        # give_path uses routePath only to populate an env var for a couple of
        # agents (if, lmdrive); a placeholder is fine since routing here goes
        # through set_route(), not a CARLA route file.
        agent_path, conf = give_path(agent_name, self.pcla_dir, routePath="pcla_bridge")
        module_dir = os.path.dirname(agent_path)
        module_key = f"pcla_dynamic_agent.{os.path.basename(agent_path).split('.')[0]}"

        original_sys_path = list(sys.path)
        if module_dir in sys.path:
            sys.path.remove(module_dir)
        sys.path.insert(0, module_dir)
        clear_agent_modules(module_key)
        try:
            spec = importlib.util.spec_from_file_location(module_key, agent_path)
            module_agent = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module_agent
            spec.loader.exec_module(module_agent)
        finally:
            sys.path = original_sys_path

        entry = getattr(module_agent, "get_entry_point")()
        self.agent = getattr(module_agent, entry)(conf)

        # Cache the agent's declared sensor specs, keyed by id, so we know how to
        # marshal each incoming reading (image vs lidar vs gnss vs imu vs speed).
        self._sensor_specs = {s["id"]: s for s in self.agent.sensors()}

        self._frame = 0
        self._sim_time = 0.0
        self._route_set = False

    # ── route ────────────────────────────────────────────────────────────────

    def sensor_layout(self):
        """Return the sensor specs the agent expects (id -> spec). Use this to see
        exactly which sensors, at which resolution/placement, you must supply."""
        return self._sensor_specs

    def set_route(self, gps_waypoints, commands=None):
        """Give the agent its global plan.

        Args:
            gps_waypoints : list of (lat, lon) or (lat, lon, alt) — WGS84 degrees.
                            This is where you intend to drive.
            commands      : optional list of high-level commands per waypoint (see
                            RoadOption). Defaults to LANEFOLLOW everywhere.

        NOTE ON FRAMES (the part that needs validation on real hardware):
        Agents match live GPS against these waypoints in a metric frame obtained
        by converting lat/lon -> local meters. Below we build the world plan with a
        simple local equirectangular projection relative to the first waypoint.
        That is internally consistent, and PCLA's GNSS sign-guard will reconcile the
        latitude/longitude sign convention automatically — but you MUST verify, with
        the vehicle stationary at a known point, that the agent's computed position
        lands where you expect before trusting any steering it produces.
        """
        from leaderboard_codes.route_manipulation import RoadOption

        if commands is None:
            commands = [RoadOption.LANEFOLLOW] * len(gps_waypoints)

        lat0 = gps_waypoints[0][0]
        R = 6378137.0                       # Earth radius (m), WGS84
        m_per_deg_lat = R * math.pi / 180.0
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))

        global_plan_gps, global_plan_world = [], []
        for wp, cmd in zip(gps_waypoints, commands):
            lat, lon = wp[0], wp[1]
            alt = wp[2] if len(wp) > 2 else 0.0
            global_plan_gps.append(({"lat": lat, "lon": lon, "z": alt}, cmd))
            # local ENU-ish meters relative to the first waypoint
            x = (lon - gps_waypoints[0][1]) * m_per_deg_lon
            y = (lat - lat0) * m_per_deg_lat
            tf = carla.Transform(carla.Location(x=x, y=y, z=alt))
            global_plan_world.append((tf, cmd))

        self.agent.set_global_plan(global_plan_gps, global_plan_world)
        self._route_set = True

    # ── per-frame step ─────────────────────────────────────────────────────────

    def step(self, sensors, dt=0.05):
        """Run one control step.

        Args:
            sensors : dict keyed by the agent's sensor ids (see sensor_layout()).
                      Provide each reading in its "natural" form; this method
                      converts to CARLA's exact wire format:
                        camera  -> HxWx3 uint8 BGR         (CARLA order; alpha added)
                        lidar   -> Nx4    float32 [x,y,z,intensity]
                        gnss    -> (lat, lon, alt)         degrees / metres
                        imu     -> (ax,ay,az, gx,gy,gz, compass_rad)
                        speed   -> float, forward speed in m/s
            dt      : seconds since previous step (advances the agent's clock).

        Returns:
            carla.VehicleControl with .throttle, .steer, .brake. Convert those to
            your actuators, or feed to your simulator's vehicle-control input. (Also
            exposes .hand_brake, .reverse if the agent set them.)
        """
        if not self._route_set:
            raise RuntimeError("Call set_route(...) before step(...).")

        input_data = self._marshal(sensors)

        # Advance the agent-visible clock (some agents read GameTime.get_time()).
        self._frame += 1
        self._sim_time += dt
        self._GameTime.on_carla_tick(_StepTimestamp(self._frame, self._sim_time, dt))
        timestamp = self._GameTime.get_time()

        # vehicle=None: sensor-only agents only use `vehicle` as the GNSS-guard's
        # calibration reference, which falls back to the route start when it's None.
        control = self.agent.run_step(input_data, timestamp, vehicle=None)
        control.manual_gear_shift = False
        return control

    # ── sensor marshalling ──────────────────────────────────────────────────────

    def _marshal(self, sensors):
        """Convert natural readings into CARLA's { id: (frame, array) } format."""
        missing = set(self._sensor_specs) - set(sensors)
        # The debug top-down 'bev' camera and the optional pseudo/event sensors
        # (opendrive map, collision) need not be supplied every step.
        missing = {m for m in missing
                   if not self._sensor_specs[m]["type"].startswith(OPTIONAL_SENSOR_TYPES)
                   and m != "bev"}
        if missing:
            raise KeyError(f"Missing sensor readings for: {sorted(missing)}. "
                           f"Agent expects: {sorted(self._sensor_specs)}")

        out = {}
        for sid, value in sensors.items():
            spec = self._sensor_specs.get(sid)
            if spec is None:
                continue  # ignore extras the agent didn't ask for
            stype = spec["type"]

            if stype == "sensor.camera.rgb":
                out[sid] = (self._frame, self._to_carla_image(value))
            elif stype == "sensor.lidar.ray_cast":
                out[sid] = (self._frame, np.asarray(value, dtype=np.float32).reshape(-1, 4))
            elif stype == "sensor.other.gnss":
                out[sid] = (self._frame, np.asarray(value, dtype=np.float64))          # [lat,lon,alt]
            elif stype == "sensor.other.imu":
                out[sid] = (self._frame, np.asarray(value, dtype=np.float64))          # 7-vector
            elif stype == "sensor.speedometer":
                out[sid] = (self._frame, {"speed": float(value)})
            else:
                out[sid] = (self._frame, value)  # radar / anything else: pass through
        return out

    @staticmethod
    def _to_carla_image(img):
        """CARLA cameras deliver HxWx4 BGRA uint8. Accept HxWx3 (assumed BGR, CARLA's
        order) or HxWx4 and normalise to BGRA. If your camera is RGB, convert to BGR
        BEFORE calling — the agents expect CARLA's BGRA and many flip BGR->RGB
        internally, so getting this wrong silently swaps the colour channels."""
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"camera image must be HxWx3 or HxWx4, got {arr.shape}")
        if arr.shape[2] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr.astype(np.uint8), alpha], axis=2)
        return np.ascontiguousarray(arr.astype(np.uint8))


# ── demo: prove the data flow with synthetic sensors (no hardware, no server) ──
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Demo: drive a PCLA agent from SYNTHETIC sensor data, no CARLA "
                    "server. It proves the data flow end-to-end (the agent loads, its "
                    "sensors are arranged, and it returns a control) using blank "
                    "frames. To actually drive, replace the synthetic block with real "
                    "readings from your source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("agent", nargs="?", default="tfv4_lav_0",
                   help="Agent to load: '<agent>_<variant>' exactly as in agents.json / "
                        "Seed-ensemble agents need their seed suffix, e.g. tfv4_lav_0.")
    p.add_argument("--frames", type=int, default=3,
                   help="Number of synthetic control steps to run in THIS demo. Each step "
                        "feeds one blank sensor frame through the agent and prints the "
                        "throttle/steer/brake it returns. It only controls how long the "
                        "demo loops — in a real deployment you call bridge.step() once per "
                        "incoming sensor frame, for as long as you're driving.")
    args = p.parse_args()

    print(f"Loading {args.agent} standalone (no CARLA server)...")
    bridge = PCLAAgentBridge(args.agent)

    print("\nThis agent expects these sensors:")
    for sid, spec in bridge.sensor_layout().items():
        extra = {k: v for k, v in spec.items() if k not in ("type", "id")}
        print(f"  {sid:16s} {spec['type']:26s} {extra}")

    # ── ROUTE ──────────────────────────────────────────────────────────────────
    # A route is just the sequence of GPS waypoints you want the agent to follow.
    # Each is (latitude, longitude) in WGS84 degrees; (lat, lon, altitude) also
    # works. It is a *global plan*, not a dense trajectory — a handful of points
    # that outline the path is enough; the agent steers between them from its
    # sensors. (These example coords trace a short path near Berlin.)
    route = [
        (52.520000, 13.404000),   # start
        (52.520450, 13.404000),   # ~50 m "north"
        (52.520900, 13.404000),   # ~100 m
        (52.521350, 13.404300),   # begins to curve right
        (52.521600, 13.404900),   # continues right
    ]
    bridge.set_route(route)

    # Optionally attach a high-level command per waypoint (default: LANEFOLLOW):
    #   from leaderboard_codes.route_manipulation import RoadOption
    #   bridge.set_route(route, commands=[
    #       RoadOption.LANEFOLLOW, RoadOption.LANEFOLLOW, RoadOption.LANEFOLLOW,
    #       RoadOption.RIGHT,      RoadOption.LANEFOLLOW])

    # >>> INTEGRATION: replace this synthetic block with sensor data from your
    # source (real vehicle, BeamNG / other simulator, or a recorded log).
    # Here the GNSS is pinned to the route start so the demo is self-consistent
    # (the "ego" sits at the first waypoint); everything else is blank.
    def synthetic_reading():
        s = {}
        for sid, spec in bridge.sensor_layout().items():
            t = spec["type"]
            if t == "sensor.camera.rgb":
                s[sid] = np.zeros((spec["height"], spec["width"], 3), dtype=np.uint8)
            elif t == "sensor.lidar.ray_cast":
                s[sid] = np.zeros((1000, 4), dtype=np.float32)
            elif t == "sensor.other.gnss":
                s[sid] = np.array([route[0][0], route[0][1], 0.0])   # at route start
            elif t == "sensor.other.imu":
                s[sid] = np.array([0, 0, 9.81, 0, 0, 0, 0.0])   # compass=0 rad
            elif t == "sensor.speedometer":
                s[sid] = 0.0
        return s

    print(f"\nRunning {args.frames} synthetic steps...")
    for i in range(args.frames):
        ctrl = bridge.step(synthetic_reading(), dt=0.05)
        print(f"  frame {i}: throttle={ctrl.throttle:.3f} steer={ctrl.steer:+.3f} brake={ctrl.brake:.3f}")

    print("\nData flow works. Now: feed your source's sensors into step(), and apply "
          "the returned throttle/steer/brake in your source (actuators, or the sim's "
          "vehicle-control input). If your sink is a real vehicle, mind the SAFETY note.")
