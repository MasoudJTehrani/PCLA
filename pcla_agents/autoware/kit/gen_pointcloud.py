#!/usr/bin/env python
# Copyright (c) 2025 Testing Automated group (TAU) at
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
"""Generate an Autoware NDT pointcloud map (.pcd) for a CARLA town.

Autoware localizes with NDT, which needs a per-town pointcloud map. Rather than downloading
one (only a few towns have public maps, and custom/source-built towns have none), this tool
BUILDS it from the running simulator: it sweeps a LiDAR over every road waypoint, accumulates
the returns in the Autoware `map` frame (CARLA world with y negated -- the same transform the
carla interface applies to live scans, so NDT matches), voxel-downsamples, and writes a PCD.

Run once per town at setup, on a bare CARLA (NO zenoh bridge running -- this tool controls
synchronous mode itself; the bridge is the sync master during normal operation and the two
must not fight). Works for ANY town, including customized / source-built maps.

Usage:
    python gen_pointcloud.py --town Town01 \
        --out /home/vortex/autoware_carla_launch/carla_map/Town01/pointcloud_map.pcd
"""
import argparse
import queue
import time

import carla
import numpy as np


def voxel_downsample(pts, voxel):
    """Keep one point per voxel cell (cheap numpy voxel grid; no PCL/open3d needed)."""
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def write_pcd_binary(path, pts):
    """Write an uncompressed binary PCD (FIELDS x y z). PCL's loader (used by NDT) reads it."""
    n = len(pts)
    header = (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
        f'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {n}\nDATA binary\n'
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(np.ascontiguousarray(pts, dtype='<f4').tobytes())


def main():
    ap = argparse.ArgumentParser(description='Build an Autoware NDT pointcloud map from CARLA.')
    ap.add_argument('--town', required=True, help='CARLA town name, e.g. Town01')
    ap.add_argument('--out', required=True, help='output .pcd path')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--step', type=float, default=8.0, help='metres between scan poses along lanes')
    ap.add_argument('--voxel', type=float, default=0.2, help='downsample voxel size (m)')
    ap.add_argument('--range', type=float, default=100.0, help='LiDAR range (m)')
    ap.add_argument('--z', type=float, default=1.8, help='LiDAR height above the waypoint (m)')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    # The zenoh bridge may have left CARLA frozen in synchronous mode (it is the sync master
    # during normal operation). Unfreeze first, or load_world/ticking would time out.
    st0 = world.get_settings()
    if st0.synchronous_mode:
        st0.synchronous_mode = False
        world.apply_settings(st0)
        try:
            world.tick()   # push the pending sync tick through so async takes effect
        except Exception:
            pass
    if args.town.lower() not in world.get_map().name.lower():
        print(f'loading {args.town} ...')
        world = client.load_world(args.town)
        time.sleep(3.0)

    orig = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    world.apply_settings(s)

    bl = world.get_blueprint_library()
    bp = bl.find('sensor.lidar.ray_cast')
    bp.set_attribute('range', str(args.range))
    bp.set_attribute('channels', '64')
    bp.set_attribute('points_per_second', '700000')
    bp.set_attribute('rotation_frequency', '20')
    bp.set_attribute('upper_fov', '20')
    bp.set_attribute('lower_fov', '-25')
    lidar = world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, 50)))
    q = queue.Queue()
    lidar.listen(q.put)

    wps = world.get_map().generate_waypoints(args.step)
    print(f'{len(wps)} scan poses (step={args.step} m); sweeping...')
    acc = np.empty((0, 3), np.float32)
    try:
        for i, wp in enumerate(wps):
            t = wp.transform
            loc = carla.Location(t.location.x, t.location.y, t.location.z + args.z)
            lidar.set_transform(carla.Transform(loc, t.rotation))
            world.tick()
            try:
                data = q.get(timeout=2.0)
            except queue.Empty:
                continue
            raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
            mat = np.array(data.transform.get_matrix())          # sensor-local -> CARLA world
            world_pts = (mat @ np.c_[raw, np.ones(len(raw))].T).T[:, :3]
            world_pts[:, 1] = -world_pts[:, 1]                   # CARLA world -> Autoware map (y flip)
            acc = np.vstack((acc, world_pts.astype(np.float32)))
            if i and i % 200 == 0:
                acc = voxel_downsample(acc, args.voxel)
                print(f'  {i}/{len(wps)} poses -> {len(acc)} pts')
    finally:
        lidar.stop()
        lidar.destroy()
        world.apply_settings(orig)

    acc = voxel_downsample(acc, args.voxel)
    print(f'final {len(acc)} pts; extent '
          f'x[{acc[:,0].min():.1f},{acc[:,0].max():.1f}] '
          f'y[{acc[:,1].min():.1f},{acc[:,1].max():.1f}] '
          f'z[{acc[:,2].min():.1f},{acc[:,2].max():.1f}]')
    write_pcd_binary(args.out, acc)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
