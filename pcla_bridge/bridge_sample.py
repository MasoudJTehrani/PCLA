"""Run a PCLA agent over a recorded drive — no CARLA server.

Reads a sample directory (the bundled samples/carla_run, or one you export from
your own application in the same layout):

    <sample>/
      route.json       # {"waypoints": [[lat,lon], ...]}
      video.mp4        # front-camera frames (one per telemetry row)
      telemetry.jsonl  # per-frame {gps:[lat,lon,alt], speed, imu:[7], dt}

feeds each frame through PCLAAgentBridge, and writes the control the agent asked
for to <out>/control.csv. With --overlay it also renders <out>/overlay.mp4 with
the throttle/steer/brake drawn on each frame, so you can *watch* the agent react.

Cameras: the single front video is resized to every forward-facing camera the
agent declares. Agents that also want side/rear cameras or LiDAR get zero-filled
placeholders for those (with a warning) — fine for a plumbing test, but such an
agent won't drive sensibly from one video. Prefer a front-camera-only agent such
as `neat_aim2dsem`.

Usage:
    python bridge_sample.py --agent neat_aim2dsem --sample samples/carla_run --overlay
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcla_bridge import PCLAAgentBridge


def load_sample(sample_dir):
    with open(os.path.join(sample_dir, "route.json")) as f:
        route = json.load(f)["waypoints"]
    telemetry = []
    with open(os.path.join(sample_dir, "telemetry.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                telemetry.append(json.loads(line))
    video_path = os.path.join(sample_dir, "video.mp4")
    return route, telemetry, video_path


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


def draw_overlay(frame_bgr, i, ctrl):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 74), (0, 0, 0), -1)
    cv2.putText(img, f"frame {i:04d}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def bar(x, label, val, lo, hi, color):
        cv2.putText(img, f"{label} {val:+.2f}", (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        frac = (val - lo) / (hi - lo)
        cv2.rectangle(img, (x, 34), (x + 150, 50), (60, 60, 60), -1)
        cv2.rectangle(img, (x, 34), (x + int(150 * max(0, min(1, frac))), 50), color, -1)

    bar(130, "thr", ctrl.throttle, 0, 1, (0, 200, 0))
    bar(330, "brk", ctrl.brake, 0, 1, (0, 0, 220))
    bar(530, "str", ctrl.steer, -1, 1, (220, 160, 0))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="neat_aim2dsem",
                    help="agent to drive (front-camera agents work best here)")
    ap.add_argument("--sample", default=os.path.join(os.path.dirname(__file__), "samples", "carla_run"),
                    help="sample directory (route.json + video.mp4 + telemetry.jsonl)")
    ap.add_argument("--out", default=None, help="output directory (default: <sample>/replay_out)")
    ap.add_argument("--overlay", action="store_true", help="also render overlay.mp4")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.sample, "replay_out")
    os.makedirs(out_dir, exist_ok=True)

    route, telemetry, video_path = load_sample(args.sample)
    print(f"Loading agent {args.agent} ...")
    bridge = PCLAAgentBridge(args.agent)
    bridge.set_route([(w[0], w[1]) for w in route])

    cap = cv2.VideoCapture(video_path)
    writer, warned = None, set()
    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(telemetry):
            break
        tele = telemetry[i]
        ctrl = bridge.step(build_sensors(bridge, frame, tele, warned), dt=tele.get("dt", 0.05))
        rows.append((i, ctrl.throttle, ctrl.steer, ctrl.brake))

        if args.overlay:
            over = draw_overlay(frame, i, ctrl)
            if writer is None:
                h, w = over.shape[:2]
                writer = cv2.VideoWriter(os.path.join(out_dir, "overlay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(over)
        i += 1

    cap.release()
    if writer is not None:
        writer.release()

    with open(os.path.join(out_dir, "control.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["frame", "throttle", "steer", "brake"])
        cw.writerows(rows)

    if rows:
        thr = np.mean([r[1] for r in rows]); steer = [r[2] for r in rows]
        print(f"\nReplayed {len(rows)} frames.")
        print(f"  throttle: mean {thr:.3f}")
        print(f"  steer   : min {min(steer):+.3f}  max {max(steer):+.3f}  mean {np.mean(steer):+.3f}")
        print(f"  -> {os.path.join(out_dir, 'control.csv')}"
              + (f"\n  -> {os.path.join(out_dir, 'overlay.mp4')}" if args.overlay else ""))
    else:
        print("No frames replayed — check that video.mp4 and telemetry.jsonl line up.")


if __name__ == "__main__":
    main()
