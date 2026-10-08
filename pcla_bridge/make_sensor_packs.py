"""Capture compact 'sensor packs' — a concrete, per-rig reference of what each
agent's input_data actually looks like, so bridge users can match their own source.

For each agent it spawns that agent's sensors in CARLA and gently drives the ego
forward for a few frames, recording the input_data the agent would receive — WITHOUT
running the model forward pass (we just read sensor_interface), so it's fast and
needs no GPU compute. Driving makes the captured values vary (speed ramps up, GPS
advances) so the pack shows realistic motion rather than a parked snapshot.

Output (small, committable), one dir per rig, named to list every agent it covers
('<base>_<variant1>+<variant2>+...', seed numbers dropped), e.g. 'tfv4_lav+wp+l6':

    samples/sensor_packs/<base>_<variants>/
      manifest.json            # per sensor: type, spec, array shape, dtype, a
                               #   units/convention note, and the actual captured
                               #   values (all frames for gnss/imu/speed; sample
                               #   rows for lidar). Lists which agents share the rig.
      <cam>_f00.jpg, _f05.jpg  # 2 example frames per camera (downscaled thumbnails;
                               #   the true array is HxWx4 BGRA uint8, see manifest)
      <lidar>_f00.txt          # a few sample points + per-frame point counts
    README.md / index.json     # top-level: every agent -> its pack (+ non-portable ones)

Agents are DEDUPED by rig: agents whose sensors() are identical share one pack.
Privileged / unsupported agents are skipped; GPU-heavy VLAs can be --exclude'd now
and appended later (the run is resumable).

Requires a running CARLA server (local or remote).
Usage:
    python make_sensor_packs.py --host 10.21.13.26 --agents neat_aim2dsem tfv4_lav_0 minddrive_05b
    python make_sensor_packs.py --host 10.21.13.26            # all agents in agents.json
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import carla
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


# Per-sensor-TYPE format notes (the format is per type; specs are per agent).
FORMAT_NOTES = {
    "sensor.camera.rgb":
        "Raw array is HxWx4 uint8 in BGRA order (CARLA camera). Feed the bridge BGR "
        "(HxWx3) or BGRA (HxWx4); it appends alpha if you pass 3 channels.",
    "sensor.lidar.ray_cast":
        "Nx4 float32 [x, y, z, intensity] in the sensor's local frame; N varies per frame.",
    "sensor.other.radar":
        "Nx4 float32, one row per detection [depth (m), altitude (rad), azimuth (rad), "
        "velocity (m/s)]; N varies per frame (velocity < 0 means closing).",
    "sensor.other.gnss":
        "[latitude, longitude, altitude] float64 (WGS84 degrees / metres).",
    "sensor.other.imu":
        "[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, compass] float64; "
        "compass in radians.",
    "sensor.speedometer":
        "forward speed in m/s; delivered as {'speed': float}.",
}

# Agent families the bridge refuses (mirrors pcla_bridge.py) — a real-sensor pack does
# not apply, so we skip them and just note why in the index. Privileged agents read
# simulator ground-truth (ego pose / BEV map); autoware is an external ROS 2 stack.
PRIVILEGED_AGENTS = {"plant", "plant2", "carl"}
UNSUPPORTED_AGENTS = {"autoware"}


def not_portable_reason(agent_name):
    """Return a short reason if this agent has no real-sensor pack, else None."""
    base = agent_name.split("_")[0]
    if base in UNSUPPORTED_AGENTS:
        return "not a run_step model (external ROS 2 stack)"
    if base in PRIVILEGED_AGENTS:
        return "privileged: reads simulator ground-truth (ego pose / BEV map)"
    return None


def _variant_label(name):
    """Agent full name -> variant label: drop the base family and any trailing seed.
    e.g. 'tfv4_lav_0' -> 'lav', 'tfv6_4cameras' -> '4cameras', 'tt_tt' -> 'tt'."""
    base = name.split("_")[0]
    rest = name[len(base) + 1:] if "_" in name else ""
    rest = re.sub(r"_\d+$", "", rest)        # strip a trailing seed like _0
    return rest or base


def pack_dirname(agents, agent_order):
    """Name a shared pack after every agent it covers, ordered as they appear in
    agents.json, with seed numbers dropped.

    Same family  -> base once + short variants: 'tfv4_lav+wp+l6'.
    Mixed family -> full (seed-stripped) names, so every agent stays findable and no
    one family's prefix hides another: 'minddrive_05b+minddrive_3b+orion_base'
    (e.g. ORION and MindDrive share one 6-camera rig)."""
    order = {a: i for i, a in enumerate(agent_order)}
    ags = sorted(agents, key=lambda a: order.get(a, 10 ** 6))
    bases = {a.split("_")[0] for a in ags}
    if len(bases) == 1:
        base = ags[0].split("_")[0]
        parts = [_variant_label(a) for a in ags]
        return f"{base}_{'+'.join(dict.fromkeys(parts))}"
    parts = [re.sub(r"_\d+$", "", a) for a in ags]        # full names, seed stripped
    return "+".join(dict.fromkeys(parts))


def finalize_pack_names(out_dir, agent_order, agent_index=None, nonportable=None):
    """Rename every pack dir to list all the agents it covers, then (re)write the index.

    Runs at the end of a capture (so newly-joined agents are reflected) and is also
    usable standalone (--finalize-only) since it can reload the index from disk. The
    per-pack manifest.json does not store its own folder name, so renaming is safe.
    """
    if agent_index is None:
        idx = json.load(open(os.path.join(out_dir, "index.json")))
        agent_index = {a: (i["pack"], i["sensors"]) for a, i in idx.get("packs", {}).items()}
        if nonportable is None:
            nonportable = idx.get("not_portable", {})
    nonportable = nonportable or {}

    groups = {}                              # current pack dir -> [agents]
    for a, (pack, _sensors) in agent_index.items():
        groups.setdefault(pack, []).append(a)

    for pack, ags in groups.items():
        new = pack_dirname(ags, agent_order)
        if new != pack:
            src, dst = os.path.join(out_dir, pack), os.path.join(out_dir, new)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)       # stale target left by an earlier run
                os.rename(src, dst)
        for a in ags:
            agent_index[a] = (new, agent_index[a][1])

    _write_index(out_dir, agent_index, nonportable)
    return agent_index


def rig_signature(sensor_specs):
    """Stable hash of a sensor configuration (ignoring ids), so identical rigs dedup."""
    canon = sorted(
        json.dumps({k: v for k, v in s.items() if k != "id"}, sort_keys=True)
        for s in sensor_specs)
    return hashlib.sha1("\n".join(canon).encode()).hexdigest()[:10]


def _capture_frames(agent_instance, world, vehicle, n_frames, warmup):
    """Drive the ego and record `n_frames` of input_data after a warmup.

    A Model3 is slow off the line and its physics settle erratically for the first
    ~10-15 ticks, so we throttle through a warmup first and only capture once it is
    actually moving (so speed / GPS / heading vary across the recorded frames).
    """
    from leaderboard_codes.timer import GameTime
    si = agent_instance.sensor_interface
    frames = []
    control = carla.VehicleControl(throttle=0.7, steer=0.0)
    for i in range(warmup + n_frames):
        vehicle.apply_control(control)
        world.tick()
        # Advance the agent-visible clock, exactly as PCLA.get_action does. The
        # pseudo-sensors (speedometer) gate their output on GameTime, so without
        # this get_data() blocks forever waiting for 'speed'.
        GameTime.on_carla_tick(world.get_snapshot().timestamp)
        data = si.get_data()                 # {id: (frame, array_or_dict)} — no forward pass
        if i >= warmup:
            frames.append(data)
    return frames


def write_pack(out_dir, rig_id, agents, specs, frames):
    os.makedirs(out_dir, exist_ok=True)
    spec_by_id = {s["id"]: s for s in specs}
    manifest = {"rig_id": rig_id, "agents_using_this_rig": sorted(agents),
                "n_frames_captured": len(frames), "sensors": {}}

    for sid, spec in spec_by_id.items():
        stype = spec["type"]
        # pull this sensor's array across frames (skip frames where it's absent)
        seq = [f[sid][1] for f in frames if sid in f]
        if not seq:
            continue
        entry = {"type": stype,
                 "spec": {k: v for k, v in spec.items() if k != "type"},
                 "format_note": FORMAT_NOTES.get(stype, "")}
        sample = seq[0]

        if stype == "sensor.camera.rgb":
            arr = np.asarray(sample)
            entry["array_shape"] = list(arr.shape)
            entry["dtype"] = str(arr.dtype)
            entry["example_images_note"] = (
                "thumbnails (downscaled JPEG) for visual reference only; the true array "
                "is array_shape/dtype above (BGRA uint8, full resolution).")
            imgs = []
            for idx in (0, min(5, len(seq) - 1)):
                bgr = seq[idx][:, :, :3]                    # BGRA/BGR -> BGR for a viewable image
                h, w = bgr.shape[:2]
                if w > 480:                                 # thumbnail: keep packs small
                    bgr = cv2.resize(bgr, (480, max(1, round(480 * h / w))))
                fn = f"{sid}_f{idx:02d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fn), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                imgs.append(fn)
            entry["example_images"] = sorted(set(imgs))
        elif stype == "sensor.lidar.ray_cast" or stype.startswith("sensor.other.radar"):
            # Variable-length point clouds (lidar) / detection lists (radar): store a
            # small sample + per-frame counts, NOT every point of every frame (that
            # bloats the manifest to hundreds of KB for a multi-radar rig).
            arr = np.asarray(sample)
            entry["array_shape"] = ["N", arr.shape[-1]]
            entry["dtype"] = str(arr.dtype)
            entry["points_per_frame"] = [int(np.asarray(s).shape[0]) for s in seq]
            hdr = ("first 8 points: x y z intensity" if stype.startswith("sensor.lidar")
                   else "first 8 detections: depth(m) altitude(rad) azimuth(rad) velocity(m/s)")
            fn = f"{sid}_f00.txt"
            np.savetxt(os.path.join(out_dir, fn), arr[:8], fmt="%.4f", header=hdr)
            entry["sample_file"] = fn
        elif stype == "sensor.speedometer":
            entry["dtype"] = "float"
            entry["values_all_frames"] = [round(float(s["speed"]), 4) for s in seq]
        else:  # gnss, imu, other numeric vectors
            arr = np.asarray(sample)
            entry["array_shape"] = list(arr.shape)
            entry["dtype"] = str(arr.dtype)
            entry["values_all_frames"] = [np.asarray(s).round(6).tolist() for s in seq]

        manifest["sensors"][sid] = entry

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--agents", nargs="+", default=None, help="agent full names; default = all in agents.json")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="agent full names to skip this run (e.g. GPU-heavy VLAs to capture "
                         "later on a free GPU); the run stays resumable so they can be appended")
    ap.add_argument("--frames", type=int, default=10, help="frames to capture per rig")
    ap.add_argument("--warmup", type=int, default=20,
                    help="frames to throttle before capturing (lets the ego reach speed "
                         "so captured values vary; a Model3 needs ~15 ticks to get moving)")
    ap.add_argument("--out", default=os.path.join(_HERE, "samples", "sensor_packs"))
    ap.add_argument("--finalize-only", action="store_true",
                    help="don't capture; just (re)name pack dirs to list every covered "
                         "agent and rewrite the index from the existing packs on disk")
    args = ap.parse_args()
    os.chdir(_ROOT)

    from pcla_functions.test_all_agents import load_agents_config
    agent_order = load_agents_config(_ROOT)[0]          # full agents.json order (for pack names)

    if args.finalize_only:
        finalize_pack_names(args.out, agent_order)
        print(f"Renamed packs to list covered agents; index: {os.path.join(args.out, 'README.md')}")
        return

    agent_list = args.agents or agent_order

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    print(f"Connecting to CARLA at {args.host}:{args.port} ...")
    client.load_world(args.town)
    world = client.get_world()
    settings = world.get_settings()
    if not settings.synchronous_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp = world.get_blueprint_library().filter("model3")[0]
    spawns = world.get_map().get_spawn_points()
    route_path = os.path.join(_ROOT, "sample_route.xml")

    from PCLA import PCLA

    exclude = set(args.exclude or [])
    seen_rigs = {}       # signature -> pack dir name (named after the first agent using it)
    agent_index = {}     # agent name -> (pack dir name, [sensor ids]) for the README index
    nonportable = {}     # agent name -> reason (privileged / unsupported; no pack)

    # Resume: fold in any packs already on disk so re-runs accumulate (lets you capture
    # the light agents now and append the GPU-heavy VLAs later without losing the index).
    idxf = os.path.join(args.out, "index.json")
    if os.path.isfile(idxf):
        _idx = json.load(open(idxf))
        for a, info in _idx.get("packs", {}).items():
            agent_index[a] = (info["pack"], info["sensors"])
        nonportable.update(_idx.get("not_portable", {}))   # keep the section on subset re-runs
    if os.path.isdir(args.out):
        for pack in os.listdir(args.out):
            mf = os.path.join(args.out, pack, "manifest.json")
            if os.path.isfile(mf):
                seen_rigs[json.load(open(mf))["rig_id"]] = pack

    try:
        for name in agent_list:
            if name in exclude:
                print(f"  {name}: excluded this run (deferred; append later)")
                continue
            reason = not_portable_reason(name)
            if reason:
                nonportable[name] = reason
                print(f"  {name}: skipped — {reason}")
                continue
            if name in agent_index and os.path.isdir(os.path.join(args.out, agent_index[name][0])):
                print(f"  {name}: already captured (pack '{agent_index[name][0]}'), skipping")
                continue
            vehicle = world.try_spawn_actor(bp, spawns[31] if len(spawns) > 31 else spawns[0])
            world.tick()
            pcla = None
            try:
                pcla = PCLA(name, vehicle, route_path, client)
                specs = pcla.agent_instance.sensors()
                sig = rig_signature(specs)
                sensor_ids = [s["id"] for s in specs]
                if sig in seen_rigs:
                    # same rig: append this agent to the existing pack's manifest; the
                    # capture (loading the model already paid for) is skipped entirely.
                    pack = seen_rigs[sig]
                    mf = os.path.join(args.out, pack, "manifest.json")
                    m = json.load(open(mf))
                    m["agents_using_this_rig"] = sorted(set(m["agents_using_this_rig"]) | {name})
                    json.dump(m, open(mf, "w"), indent=2)
                    print(f"  {name}: same sensors as '{pack}' (shares its pack)")
                else:
                    # new rig: drive + capture, pack dir named after THIS agent (recognizable)
                    frames = _capture_frames(pcla.agent_instance, world, vehicle,
                                             args.frames, args.warmup)
                    pack = name
                    seen_rigs[sig] = pack
                    m = write_pack(os.path.join(args.out, pack), sig, [name], specs, frames)
                    print(f"  {name}: new pack '{pack}'  ({len(m['sensors'])} sensors: "
                          f"{', '.join(m['sensors'])})")
                agent_index[name] = (seen_rigs[sig], sensor_ids)
            except (Exception, KeyboardInterrupt) as e:
                # KeyboardInterrupt is how the PCLA watchdog signals a load timeout;
                # catch it too so one stuck agent can't abort the whole sweep.
                print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:80]}")
            finally:
                if pcla is not None:
                    pcla.cleanup()          # destroys the vehicle too
                elif vehicle is not None and vehicle.is_alive:
                    vehicle.destroy()

        finalize_pack_names(args.out, agent_order, agent_index, nonportable)
        print(f"\nWrote {len(seen_rigs)} pack(s) for {len(agent_index)} agents to {args.out}/")
        if nonportable:
            print(f"Skipped {len(nonportable)} non-portable agent(s) (listed in index).")
        if exclude:
            print(f"Deferred {len(exclude)} excluded agent(s): {', '.join(sorted(exclude))}")
        print(f"Index (agent -> pack): {os.path.join(args.out, 'README.md')}")
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)


def _write_index(out_dir, agent_index, nonportable=None):
    """Top-level README mapping EVERY agent to its pack, so users look up by name."""
    nonportable = nonportable or {}
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        "# Sensor packs — what `input_data` each agent expects",
        "",
        "Each pack is a real capture of an agent's per-frame sensor input: the sensor "
        "list, array shapes/dtypes, unit conventions, example camera frames, and the "
        "actual captured values. Use it to see exactly what to feed the bridge.",
        "",
        "Agents with **identical sensor rigs share one pack** (to save space); each pack "
        "folder is named to list every variant it covers (seed numbers dropped). Find your "
        "agent below:",
        "",
        "> **Declared vs. used.** A pack lists every sensor the agent *declares* (its "
        "`sensors()`), which for a few variants includes sensors the loaded checkpoint "
        "ignores. In particular `tfv6_noradar` and `tfv6_visiononly` still declare 4 "
        "radars (TransFuser-v6 hardcodes `radar=True` in `sensors()`), but their configs "
        "set `use_radar_detection=False`, so the model drops the radar branch. The bridge "
        "auto-fills those radars with empty “no detections” frames, so a radar-less "
        "source can still drive them; radar-using variants keep radar required.",
        "",
        "| agent | sensor pack | sensors |",
        "|---|---|---|",
    ]
    for agent in sorted(agent_index):
        pack, sensors = agent_index[agent]
        shared = "" if pack == agent else f" _(shares `{pack}`)_"
        lines.append(f"| `{agent}` | [{pack}/]({pack}/){shared} | {', '.join(sensors)} |")
    lines.append("")
    if nonportable:
        lines += [
            "## Not portable — no sensor pack",
            "",
            "These agents cannot be driven from a real sensor source (the bridge refuses "
            "them too), so there is no pack to capture:",
            "",
            "| agent | why |",
            "|---|---|",
        ]
        lines += [f"| `{a}` | {nonportable[a]} |" for a in sorted(nonportable)]
        lines.append("")
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    # machine-readable too
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"packs": {a: {"pack": p, "sensors": s} for a, (p, s) in agent_index.items()},
                   "not_portable": nonportable}, f, indent=2)


if __name__ == "__main__":
    main()
