# Copyright (c) 2025 Testing Automated group (TAU) at
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
"""PCLA adapter for the Autoware autonomous-driving stack.

Unlike every other PCLA agent, "autoware" is not an in-process model. It is a
thin bridge to an external ROS 2 Autoware stack running in Docker, connected to
CARLA by the evshary/autoware_carla_launch Zenoh bridge (jazzy branch:
CARLA 0.9.16 / Autoware 1.8.0). See the project memory `autoware-carla-phase1-proven`.

How it works (no ROS on the host):
  * PCLA owns the ego. The ego MUST be spawned with role_name "autoware_<vehicle_role>"
    (e.g. "autoware_v1"); the running zenoh_carla_bridge attaches to it by that name.
  * This adapter spawns the exact Autoware sensor kit (LiDAR "top", IMU "tamagawa",
    GNSS "ublox", camera "traffic_light") onto the ego so the bridge feeds Autoware.
    `sensors()` therefore returns [] (we do NOT use PCLA's sensor-spawn path, which
    cannot set the per-sensor role_name the bridge keys on).
  * Data-plane (per tick): the bridge writes VehicleControl straight to the CARLA
    ego, so `run_step` just reads it back with `vehicle.get_control()`.
  * Control-plane (once, at start): set goal + engage autonomous mode via
    `docker exec <container> ros2 ...` on a background thread.

Requirements: CARLA (Town01) + zenoh_carla_bridge + Autoware are already running
(the bridge sets CARLA to synchronous mode and is the tick master, so PCLA must
not tick or change the sync setting), and the host user is in the `docker` group.
"""

import math
import subprocess
import threading
import time

import carla
import yaml

from leaderboard_codes import autonomous_agent2 as autonomous_agent


def get_entry_point():
    return "AutowareAgent"


def _brake_control():
    c = carla.VehicleControl()
    c.throttle = 0.0
    c.steer = 0.0
    c.brake = 1.0
    c.hand_brake = False
    c.manual_gear_shift = False
    return c


def _spawn_autoware_sensor_kit(world, vehicle):
    """Spawn the carla_sensor_kit onto `vehicle`, mirroring evshary's carla_agent
    exactly (sensor role_name == Autoware sensor name; the bridge keys on it)."""
    bl = world.get_blueprint_library()
    spawned = []

    # LiDAR "top"  (z=2.4, yaw=270)
    lidar = bl.find("sensor.lidar.ray_cast")
    lidar.set_attribute("role_name", "top")
    for k, v in {
        "range": "100", "rotation_frequency": "20", "channels": "64",
        "upper_fov": "10", "lower_fov": "-30", "points_per_second": "1200000",
        "atmosphere_attenuation_rate": "0.004", "dropoff_general_rate": "0.45",
        "dropoff_intensity_limit": "0.8", "dropoff_zero_intensity": "0.4",
    }.items():
        lidar.set_attribute(k, v)
    spawned.append(world.spawn_actor(
        lidar, carla.Transform(carla.Location(0.0, 0.0, 2.4), carla.Rotation(yaw=270.0)),
        attach_to=vehicle))

    # IMU "tamagawa"  (z=2.4, yaw=270)
    imu = bl.find("sensor.other.imu")
    imu.set_attribute("role_name", "tamagawa")
    for a in ("noise_accel_stddev_x", "noise_accel_stddev_y", "noise_accel_stddev_z",
              "noise_gyro_stddev_x", "noise_gyro_stddev_y", "noise_gyro_stddev_z"):
        imu.set_attribute(a, "0.0")
    spawned.append(world.spawn_actor(
        imu, carla.Transform(carla.Location(0.0, 0.0, 2.4), carla.Rotation(yaw=270.0)),
        attach_to=vehicle))

    # GNSS "ublox"  (z=2.4)
    gnss = bl.find("sensor.other.gnss")
    gnss.set_attribute("role_name", "ublox")
    for a in ("noise_alt_stddev", "noise_lat_stddev", "noise_lon_stddev",
              "noise_alt_bias", "noise_lat_bias", "noise_lon_bias"):
        gnss.set_attribute(a, "0.0")
    spawned.append(world.spawn_actor(
        gnss, carla.Transform(carla.Location(0.0, 0.0, 2.4)), attach_to=vehicle))

    # RGB camera "traffic_light"  (forward; traffic-light recognition is off by default)
    cam = bl.find("sensor.camera.rgb")
    cam.set_attribute("role_name", "traffic_light")
    ext = vehicle.bounding_box.extent
    spawned.append(world.spawn_actor(
        cam, carla.Transform(carla.Location(x=0.8 * ext.x, y=0.0, z=1.3 * ext.z)),
        attach_to=vehicle))

    return spawned


class AutowareAgent(autonomous_agent.AutonomousAgent):

    # ---- lifecycle -------------------------------------------------------
    def setup(self, path_to_conf_file, route_index=None):
        with open(path_to_conf_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
        self.container = cfg.get("autoware_container", "aw_autoware")
        self.vehicle_role = cfg.get("vehicle_role", "v1")
        self.loc_timeout = float(cfg.get("localization_timeout_s", 40))
        self.goal_ahead_m = cfg.get("goal_ahead_m", None)
        self.goal_override = cfg.get("goal_override", None)
        # Optional ordered via-points [[x, y, yaw_deg], ...] in the Autoware map frame. The route
        # is forced to pass through them, in order, on the way to the goal (shape the PATH, not
        # just the destination). Autoware still plans lane-by-lane between consecutive points.
        self.checkpoints = cfg.get("checkpoints", []) or []
        self.traffic_lights = bool(cfg.get("traffic_lights", True))
        # None -> derive per-town at runtime from the loaded map:
        #   external/zenoh_autoware_v2x/carla_maps/<Town>/map_info.json
        self.tl_map_info = cfg.get("traffic_light_map_info", None)

        self._world = None
        self._sensors = []
        self._route_goal = None      # (x_aw, y_aw, yaw_aw_rad) from PCLA route endpoint
        self._engaged = False
        self._thread = None
        self._lock = threading.Lock()

    def sensors(self):
        # We spawn the Autoware kit ourselves (see module docstring); PCLA spawns nothing.
        return []

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        # Destination = last world-coord route point, CARLA -> Autoware map frame (y flip).
        try:
            last = global_plan_world_coord[-1][0]
            self._route_goal = (last.location.x, -last.location.y,
                                -math.radians(last.rotation.yaw))
        except Exception:
            self._route_goal = None

    def run_step(self, input_data, timestamp, vehicle=None):
        if vehicle is None:
            return _brake_control()

        with self._lock:
            if self._world is None:                      # one-time lazy init (we now have the ego)
                self._world = vehicle.get_world()
                self._sensors = _spawn_autoware_sensor_kit(self._world, vehicle)
                self._thread = threading.Thread(
                    target=self._bring_up_autonomous, args=(vehicle,), daemon=True)
                self._thread.start()

        # Always echo the control the bridge applied to the ego (Autoware commands STOP
        # before engage and the drive command after), so PCLA never fights the bridge.
        ctrl = vehicle.get_control()
        ctrl.manual_gear_shift = False
        return ctrl

    def destroy(self):
        # Best-effort: stop + clear the route so the stack is reusable for the next run.
        for inner in ("ros2 service call /api/operation_mode/change_to_stop "
                      "autoware_adapi_v1_msgs/srv/ChangeOperationMode '{}'",
                      "ros2 service call /api/routing/clear_route "
                      "autoware_adapi_v1_msgs/srv/ClearRoute '{}'"):
            try:
                self._ros2(inner, timeout=15)
            except Exception:
                pass
        # NB: do NOT destroy the sensors here. PCLA.cleanup() already destroys every
        # world sensor + the ego; destroying them here too caused a harmless-but-noisy
        # "failed to destroy actor: not found" on the second pass. Just drop refs.
        self._sensors = []

    # ---- traffic lights (ground-truth injection) ------------------------
    def _start_traffic_light_bridge(self):
        """(Re)start the ground-truth traffic-light injector in the container.

        Autoware's camera classifier is launched disabled (it returns UNKNOWN on CARLA's
        rendered lights); carla_gt_bridge instead mirrors CARLA's real light states into
        /perception/traffic_light_recognition/traffic_signals so the planner obeys them.
        Idempotent: kills any previous instance first so there is never a second publisher.
        """
        # Per-town traffic-light table: explicit config override, else derive from the loaded map.
        town = self._world.get_map().name.split("/")[-1]   # "Carla/Maps/Town02" -> "Town02"
        map_info = self.tl_map_info or (
            "external/zenoh_autoware_v2x/carla_maps/%s/map_info.json" % town)
        subprocess.run(["docker", "exec", self.container, "pkill", "-9", "-f",
                        "carla_gt_bridge"], capture_output=True)
        cmd = (
            "cd ~/autoware_carla_launch && source env.sh >/dev/null 2>&1 && "
            "uv run --project external/zenoh_autoware_v2x "
            "external/zenoh_autoware_v2x/carla_gt_bridge/main.py "
            "-v %s --map-info %s > /tmp/carla_gt_bridge.log 2>&1"
            % (self.vehicle_role, map_info)
        )
        subprocess.run(["docker", "exec", "-d", "-u", "aw", self.container, "bash", "-lc",
                        cmd], capture_output=True)
        print("[autoware] traffic-light injection started (ground-truth from CARLA).")

    # ---- control plane (docker exec -> ros2; no ROS on host) -------------
    def _ros2(self, inner_cmd, timeout=30):
        script = (
            "source /opt/ros/jazzy/setup.bash >/dev/null 2>&1; "
            "source /opt/autoware/setup.bash >/dev/null 2>&1; "
            "cd ~/autoware_carla_launch && source env.sh >/dev/null 2>&1 && "
            "source install/setup.bash >/dev/null 2>&1; " + inner_cmd
        )
        return subprocess.run(
            ["docker", "exec", "-u", "aw", self.container, "bash", "-lc", script],
            capture_output=True, text=True, timeout=timeout,
        )

    def _localization_ready(self):
        r = self._ros2("timeout 5 ros2 topic echo --once "
                       "/api/localization/initialization_state 2>/dev/null | grep -E '^state:'")
        return "state: 3" in r.stdout

    def _operation_autonomous(self):
        r = self._ros2("timeout 5 ros2 topic echo --once "
                       "/api/operation_mode/state 2>/dev/null | grep -E '^mode:'")
        return "mode: 2" in r.stdout

    def _autonomous_available(self):
        r = self._ros2("timeout 5 ros2 topic echo --once /api/operation_mode/state "
                       "2>/dev/null | grep is_autonomous_mode_available")
        return "true" in r.stdout.lower()

    def _routing_set(self):
        r = self._ros2("timeout 5 ros2 topic echo --once /api/routing/state "
                       "2>/dev/null | grep '^state:'")
        return "state: 2" in r.stdout or "state: 3" in r.stdout

    def _resolve_goal(self, vehicle):
        """Return (x_aw, y_aw, yaw_aw_rad). Priority: explicit override > drive-ahead > route."""
        if self.goal_override:
            x, y, yaw_deg = self.goal_override
            return float(x), float(y), math.radians(float(yaw_deg))
        if self.goal_ahead_m:
            g = self._goal_ahead(vehicle, float(self.goal_ahead_m))
            if g is not None:
                return g
            print("[autoware] drive-ahead goal not found; falling back to route endpoint.")
        return self._route_goal

    def _goal_ahead(self, vehicle, dist, min_dist=15.0):
        """Robust 'N metres ahead' goal on a lane. Tries next() and previous() over a
        distance ladder, keeps candidates farther than min_dist, and picks the one best
        aligned with the ego heading (so we always drive FORWARD, never a 0-length route)."""
        m = self._world.get_map()
        tf = vehicle.get_transform()
        wp = m.get_waypoint(tf.location)
        if wp is None:
            return None
        fwd = tf.get_forward_vector()
        best = None  # (alignment, transform)
        for d in (dist, dist * 0.75, dist * 0.5, 30.0, 20.0):
            for cw in list(wp.next(d)) + list(wp.previous(d)):
                ct = cw.transform
                dx, dy = ct.location.x - tf.location.x, ct.location.y - tf.location.y
                dd = math.hypot(dx, dy)
                if dd < min_dist:
                    continue
                align = (dx * fwd.x + dy * fwd.y) / (dd + 1e-6)  # cos(angle to ego forward)
                if best is None or align > best[0]:
                    best = (align, ct)
            if best is not None and best[0] > 0.5:  # a clearly-forward goal at this distance
                break
        if best is None:
            return None
        ct = best[1]
        return ct.location.x, -ct.location.y, -math.radians(ct.rotation.yaw)

    def _set_initial_pose(self, vehicle):
        """(Re)initialize NDT localization at the ego's actual pose. A respawned ego
        otherwise leaves localization stuck at the PREVIOUS run's final pose, so a new
        goal looks already-reached (routing -> ARRIVED) and the car never drives."""
        tf = vehicle.get_transform()
        x, y = tf.location.x, -tf.location.y
        yaw = -math.radians(tf.rotation.yaw)
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        cov = ("[0.25,0.0,0.0,0.0,0.0,0.0, 0.0,0.25,0.0,0.0,0.0,0.0, "
               "0.0,0.0,0.01,0.0,0.0,0.0, 0.0,0.0,0.0,0.01,0.0,0.0, "
               "0.0,0.0,0.0,0.0,0.01,0.0, 0.0,0.0,0.0,0.0,0.0,0.0685]")
        pose = ("{header: {frame_id: map}, pose: {pose: {position: {x: %.3f, y: %.3f, "
                "z: 0.0}, orientation: {x: 0.0, y: 0.0, z: %.6f, w: %.6f}}, "
                "covariance: %s}}" % (x, y, qz, qw, cov))
        self._ros2("ros2 topic pub -w 1 -t 5 /initialpose "
                   "geometry_msgs/msg/PoseWithCovarianceStamped \"%s\" >/dev/null 2>&1"
                   % pose, timeout=30)

    def _push_goal(self, goal):
        x, y, yaw = goal
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        pose = ("{header: {frame_id: map}, pose: {position: {x: %.3f, y: %.3f, z: 0.0}, "
                "orientation: {x: 0.0, y: 0.0, z: %.6f, w: %.6f}}}" % (x, y, qz, qw))
        # -w 1 waits until the routing adaptor's subscription is discovered before
        # publishing, so the goal is not dropped (rapid -t 3 without -w gets lost).
        self._ros2("ros2 topic pub -w 1 -t 5 /planning/mission_planning/goal "
                   "geometry_msgs/msg/PoseStamped \"%s\" >/dev/null 2>&1" % pose, timeout=30)

    def _push_checkpoint(self, cp):
        """Publish one intermediate via-point (Autoware map frame, yaw in radians).

        Checkpoints are sent AFTER the goal, in order; the mission planner re-plans the route to
        pass through them (ego -> cp1 -> ... -> goal). Same PoseStamped topic pattern as the goal.
        """
        x, y, yaw = cp
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        pose = ("{header: {frame_id: map}, pose: {position: {x: %.3f, y: %.3f, z: 0.0}, "
                "orientation: {x: 0.0, y: 0.0, z: %.6f, w: %.6f}}}" % (x, y, qz, qw))
        self._ros2("ros2 topic pub -w 1 -t 5 /planning/mission_planning/checkpoint "
                   "geometry_msgs/msg/PoseStamped \"%s\" >/dev/null 2>&1" % pose, timeout=30)

    def _bring_up_autonomous(self, vehicle):
        # 0) verify we can reach Autoware via docker (every docker exec fails silently if
        #    this process is not in the 'docker' group). sample_autoware.py normally
        #    re-execs under `sg docker` to avoid this; guard anyway with a clear message.
        if "pcla_ok" not in self._ros2("echo pcla_ok", timeout=15).stdout:
            print("[autoware] ERROR: 'docker exec %s' failed (permission denied?).\n"
                  "  Run with docker access:  sg docker -c \"python sample_autoware.py\"\n"
                  "  or log out of your desktop and back in once (joins the 'docker' group)."
                  % self.container)
            return
        # 0a) start ground-truth traffic-light injection (Autoware's camera classifier is
        #     disabled; this makes the ego obey CARLA's real red/green lights).
        if self.traffic_lights:
            self._start_traffic_light_bridge()
        # 0b) (re)initialize localization at the ego's actual pose (fixes a respawned ego
        #     inheriting the previous run's localization -> goal looks already reached).
        print("[autoware] initializing localization at the ego pose...")
        self._set_initial_pose(vehicle)
        time.sleep(4.0)
        # 1) wait for NDT localization to initialize
        loc_ok = False
        t0 = time.time()
        while time.time() - t0 < self.loc_timeout:
            if self._localization_ready():
                loc_ok = True
                break
            time.sleep(1.0)
        if not loc_ok:
            print("[autoware] WARNING: localization did not initialize in %.0fs; aborting."
                  % self.loc_timeout)
            return
        # 2) set goal
        goal = self._resolve_goal(vehicle)
        if goal is None:
            print("[autoware] No goal (no route / goal_ahead_m / goal_override); staying stopped.")
            return
        print("[autoware] localization ready; goal(map) = (%.2f, %.2f, %.1f deg)"
              % (goal[0], goal[1], math.degrees(goal[2])))
        # 3) push the goal, retrying until routing accepts it. The routing adaptor may
        #    not be subscribed the instant localization initializes, so a single publish
        #    can be silently dropped (routing stays UNSET -> no trajectory -> no engage).
        routed = False
        for _ in range(12):
            self._push_goal(goal)
            time.sleep(2.0)
            if self._routing_set():
                routed = True
                break
        if not routed:
            print("[autoware] WARNING: routing never accepted the goal; aborting engage.")
            return
        # 3b) optional via-points: publish each checkpoint (in order) so the route is forced to
        #     pass through them on the way to the goal. Lets the user choose the PATH.
        for i, cp in enumerate(self.checkpoints):
            try:
                x, y, yaw_deg = cp
                self._push_checkpoint((float(x), float(y), math.radians(float(yaw_deg))))
                print("[autoware] checkpoint %d/%d added (%.1f, %.1f)."
                      % (i + 1, len(self.checkpoints), float(x), float(y)))
                time.sleep(1.5)
            except Exception as e:
                print("[autoware] WARNING: bad checkpoint %r (%s); skipping." % (cp, e))
        print("[autoware] route set; waiting for autonomous availability...")
        # 4) wait for the planner to make autonomous mode available (trajectory ready)
        t0 = time.time()
        while time.time() - t0 < 25.0:
            if self._autonomous_available():
                break
            time.sleep(1.0)
        else:
            print("[autoware] WARNING: autonomous mode never became available "
                  "(no valid trajectory to the goal?).")
        # 5) engage autonomous, with retries, and confirm
        for _ in range(6):
            self._ros2("ros2 service call /api/operation_mode/change_to_autonomous "
                       "autoware_adapi_v1_msgs/srv/ChangeOperationMode '{}' >/dev/null 2>&1", timeout=20)
            time.sleep(1.5)
            if self._operation_autonomous():
                self._engaged = True
                print("[autoware] autonomous engaged; ego is now driving.")
                return
        print("[autoware] WARNING: engage not confirmed; check the Autoware stack.")
