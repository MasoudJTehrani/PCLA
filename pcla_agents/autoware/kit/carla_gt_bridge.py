#!/usr/bin/env python
# Copyright (c) 2025 Testing Automated group (TAU) at
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
"""Mirror CARLA's ground-truth traffic-light state into Autoware (PCLA integration).

Autoware's camera-based traffic-light classifier returns UNKNOWN on CARLA's rendered
lights, so the ego never obeys them. This node is the fix: it is READ-ONLY on CARLA
(it never calls set_state/freeze -- unlike zenoh_autoware_v2x's intersection_manager,
which HIJACKS the lights for an emergency-vehicle scenario). It reads each traffic
light's real state and republishes it, at a fixed rate, to

    <scope>/perception/traffic_light_recognition/traffic_signals

which the zenoh-bridge-ros2dds relays into Autoware as
/perception/traffic_light_recognition/traffic_signals -- the exact topic the behavior
planner consumes. This mirrors a real V2I (infrastructure-to-vehicle) deployment and is
the only traffic-light path that works in CARLA. Run Autoware with
use_traffic_light_recognition:=false so the broken camera classifier does not also
publish to that topic.

Runs inside the aw_autoware container (like v2x_light) via the zenoh_autoware_v2x uv env:
    uv run --project external/zenoh_autoware_v2x \
        external/zenoh_autoware_v2x/carla_gt_bridge/main.py -v v1 --map-info <map_info.json>

The per-town light table (map_info.json) maps each Autoware traffic_light_group_id to a
position; we resolve it to the nearest CARLA light actor at startup (positions survive
across sessions, actor ids do not). See gen_map_info.py to auto-generate it for any town.
"""
import argparse
import json
import logging
import os
import time

import carla
import zenoh
from zenoh_ros_type.autoware_msgs.autoware_perception_msgs import (
    TrafficLightElement,
    TrafficLightGroup,
    TrafficLightGroupArray,
)
from zenoh_ros_type.rcl_interfaces import Time

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

SET_TRAFFIC_SIGNALS_KEY_EXPR = '/perception/traffic_light_recognition/traffic_signals'

# Max distance (m) between a map_info.json position and the nearest CARLA light actor.
TRAFFIC_LIGHT_MATCH_THRESHOLD_M = 5.0

# str(carla.TrafficLightState) -> Autoware TrafficLightElement colour (same table as v2x_light).
CARLA_TO_COLOR = {
    'Red': TrafficLightElement.COLOR.RED.value,
    'Yellow': TrafficLightElement.COLOR.AMBER.value,
    'Green': TrafficLightElement.COLOR.GREEN.value,
    'Off': TrafficLightElement.COLOR.UNKNOWN.value,
    'Unknown': TrafficLightElement.COLOR.UNKNOWN.value,
}


def load_group_positions(map_info_path):
    """Return {autoware_group_id: (carla_x, carla_y)} from map_info.json.

    map_info.json stores traffic_light_position in the Autoware frame (y = -carla_y),
    so we negate y back into CARLA's frame to match actor world coordinates.
    """
    with open(map_info_path) as f:
        data = json.load(f)
    positions = {}
    for _section, entries in data['intersections'].items():
        for entry in entries:
            gid = int(entry['autoware_traffic_light'])
            pos = entry['traffic_light_position']
            positions[gid] = (float(pos['x']), -float(pos['y']))
    return positions


def wait_for_traffic_lights(world, attempts=20):
    """Return the traffic-light actors, waiting for a world snapshot first.

    CARLA runs in synchronous mode here (the zenoh bridge is the tick master), so a
    freshly-connected client's actor list is empty until it observes at least one tick.
    """
    for _ in range(attempts):
        try:
            world.wait_for_tick(2.0)
        except RuntimeError:
            pass
        actors = list(world.get_actors().filter('traffic.traffic_light'))
        if actors:
            return actors
    return []


def match_lights(world, group_positions):
    """Resolve {group_id: carla_light_actor} by nearest position (cf. intersection_manager)."""
    actors = wait_for_traffic_lights(world)
    if not actors:
        raise RuntimeError('no traffic.traffic_light actors in the CARLA world')
    mapping = {}
    for gid, (x, y) in group_positions.items():
        best = min(actors, key=lambda a, x=x, y=y: (a.get_location().x - x) ** 2 + (a.get_location().y - y) ** 2)
        loc = best.get_location()
        dist = ((loc.x - x) ** 2 + (loc.y - y) ** 2) ** 0.5
        if dist > TRAFFIC_LIGHT_MATCH_THRESHOLD_M:
            logging.warning(f'group {gid}: nearest CARLA light id={best.id} is {dist:.2f} m away (threshold {TRAFFIC_LIGHT_MATCH_THRESHOLD_M} m) -- check map_info.json')
        else:
            logging.debug(f'group {gid} -> CARLA light id={best.id} (dist={dist:.2f} m)')
        mapping[gid] = best
    return mapping


def build_message(group_ids, mapping):
    """Snapshot every mapped light's current CARLA state into a TrafficLightGroupArray."""
    now = time.time()
    stamp = Time(sec=int(now), nanosec=int((now - int(now)) * 1e9))
    groups = []
    for gid in group_ids:
        color = CARLA_TO_COLOR.get(str(mapping[gid].get_state()), TrafficLightElement.COLOR.UNKNOWN.value)
        groups.append(
            TrafficLightGroup(
                traffic_light_group_id=gid,
                elements=[
                    TrafficLightElement(
                        color=color,
                        shape=TrafficLightElement.SHAPE.CIRCLE.value,
                        status=TrafficLightElement.STATUS.SOLID_ON.value,
                        confidence=1.0,
                    )
                ],
                predictions=[],
            )
        )
    return TrafficLightGroupArray(stamp=stamp, traffic_light_groups=groups)


def main():
    parser = argparse.ArgumentParser(prog='carla_gt_bridge')
    parser.add_argument('-v', '--vehicle', default='v1', help='Zenoh scope; must match the ros2dds bridge -n /<scope>')
    default_map_info = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../map_info.json')
    parser.add_argument('--map-info', default=default_map_info, help='Path to map_info.json')
    parser.add_argument('--carla-host', default='localhost')
    parser.add_argument('--carla-port', type=int, default=2000)
    parser.add_argument('-e', '--connect', action='append', help='Zenoh endpoint(s) to connect to (default: multicast discovery)')
    args = parser.parse_args()

    group_positions = load_group_positions(args.map_info)
    if not group_positions:
        logging.info('[carla_gt_bridge] map_info has no traffic-light groups '
                     '(this town\'s Lanelet2 map has no signals); nothing to inject. Exiting.')
        return

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(10.0)
    world = client.get_world()
    mapping = match_lights(world, group_positions)
    group_ids = sorted(mapping)
    logging.info(f'[carla_gt_bridge] mirroring {len(group_ids)} traffic lights -> Autoware (scope={args.vehicle})')

    zenoh.init_log_from_env_or('error')
    config = zenoh.Config()
    if args.connect:
        config.insert_json5('connect/endpoints', json.dumps(args.connect))
    key = args.vehicle + SET_TRAFFIC_SIGNALS_KEY_EXPR  # ros2dds layout: <scope>/<topic>

    with zenoh.open(config) as session:
        publisher = session.declare_publisher(key)
        logging.info(f"[carla_gt_bridge] publishing on '{key}' (per CARLA tick). Ctrl+C to stop.")
        try:
            while True:
                # CARLA is synchronous (the zenoh bridge is the tick master). A client's
                # actor snapshot -- including traffic_light.get_state() -- only advances when
                # it observes a tick, so we MUST wait_for_tick each loop or we would keep
                # republishing the light states captured at startup (stale). This paces us at
                # the sim tick rate (~20 Hz), which is plenty for traffic-light state.
                try:
                    world.wait_for_tick(2.0)
                except RuntimeError:
                    time.sleep(0.1)   # frozen/async gap: republish last-known state, don't busy-spin
                publisher.put(build_message(group_ids, mapping).serialize())
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
