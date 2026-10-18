#!/usr/bin/env python
"""Minimal offline-replay demo: drive a PCLA agent from a RECORDED log, no CARLA server.

    python bridge_sample.py neat_aim2dsem samples/nuscenes_run   # real-world data
    python bridge_sample.py neat_aim2dsem samples/carla_run      # in-domain CARLA

A log is just three files: route.json ({"waypoints":[[lat,lon],...]}),
video.mp4 (front-camera frames), and telemetry.jsonl (one row per frame:
{"frame":<i>, "gps":[lat,lon,alt], "imu":[7], "speed":<m/s>, "dt":<s>}).

A fuller version of this script (overlay-video rendering, control.csv export, summary
stats) is archived with the evaluation artifacts as `bridge_sample_full.py`:
https://huggingface.co/datasets/MasoudJTehrani/pcla-bridge-evaluations
"""
import json
import sys

import cv2
import numpy as np

from pcla_bridge import PCLAAgentBridge          # the tool


def build_sensors(bridge, frame_bgr, tele, warned):
    """Map one video frame + one telemetry row onto the agent's declared sensors."""
    sensors = {}
    for sid, spec in bridge.sensor_layout().items():
        t = spec["type"]
        if t == "sensor.camera.rgb":
            # forward cameras (yaw ~0) get the real frame, resized to their spec;
            # non-forward cameras get the same frame as a stand-in (no side video here).
            resized = cv2.resize(frame_bgr, (spec["width"], spec["height"]))
            if abs(spec.get("yaw", 0.0)) > 15 and "sidecam" not in warned:
                print(f"  note: agent wants non-forward camera '{sid}' (yaw="
                      f"{spec.get('yaw')}); reusing the front frame as a placeholder.")
                warned.add("sidecam")
            sensors[sid] = resized                       # BGR, as CARLA delivers
        elif t == "sensor.lidar.ray_cast":
            if "lidar" not in warned:
                print(f"  note: agent wants LiDAR '{sid}'; feeding an empty cloud "
                      f"(this agent won't drive well from video alone).")
                warned.add("lidar")
            sensors[sid] = np.zeros((1, 4), dtype=np.float32)
        elif t == "sensor.other.gnss":
            sensors[sid] = np.asarray(tele["gps"], dtype=np.float64)
        elif t == "sensor.other.imu":
            sensors[sid] = np.asarray(tele["imu"], dtype=np.float64)
        elif t == "sensor.speedometer":
            sensors[sid] = float(tele["speed"])
    return sensors


def main():
    agent = sys.argv[1] if len(sys.argv) > 1 else "neat_aim2dsem"
    log   = sys.argv[2] if len(sys.argv) > 2 else "samples/nuscenes_run"

    # 1) Load the agent standalone -- NO CARLA server is started.
    bridge = PCLAAgentBridge(agent)

    # 2) Set the route (list of [lat, lon] GPS waypoints).
    bridge.set_route(json.load(open(f"{log}/route.json"))["waypoints"])

    # 3) Replay: one video frame + one telemetry row per step -> one control each.
    cap, warned = cv2.VideoCapture(f"{log}/video.mp4"), set()
    for line in open(f"{log}/telemetry.jsonl"):
        tele = json.loads(line)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        sensors = build_sensors(bridge, frame_bgr, tele, warned)   # resize cameras, zero-fill LiDAR
        ctrl = bridge.step(sensors, dt=tele["dt"])                 # calls agent.run_step() in-process
        print(f"frame {tele['frame']:4d}  throttle={ctrl.throttle:.2f}  "
              f"steer={ctrl.steer:+.2f}  brake={ctrl.brake:.2f}")


if __name__ == "__main__":
    main()
