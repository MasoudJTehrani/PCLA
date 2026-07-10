#!/usr/bin/env python
# Copyright (c) 2025 Testing Automated group (TAU) at
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
"""Generate the per-town traffic-light table (map_info.json) that carla_gt_bridge needs.

In a Lanelet2 map (.osm), every regulatory element of subtype `traffic_light` is one Autoware
traffic-light group -- and ITS RELATION ID is the `traffic_light_group_id` the planner keys on
(verified against Town01's hand-made table: all group ids are relation ids, not way ids). Its
position is the centroid of the traffic_light way(s) it refers to, already in the Autoware `map`
frame (local_x/local_y). carla_gt_bridge matches each group to the nearest CARLA light actor at
runtime, so this tool needs only the .osm -- no CARLA connection.

Verified on Town01: reproduces all 21 hand-made groups; positions within ~3 m (< the 5 m match
threshold). Works for any town's Lanelet2 map.

Usage:
    python gen_map_info.py --osm carla_map/Town02/lanelet2_map.osm \
        --out external/zenoh_autoware_v2x/carla_maps/Town02/map_info.json
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET


def parse_traffic_lights(osm_path):
    """Return [(group_id, x, y)] for every traffic_light regulatory element (Autoware frame)."""
    root = ET.parse(osm_path).getroot()

    nodes = {}
    for n in root.findall('node'):
        lx = ly = None
        for t in n.findall('tag'):
            if t.get('k') == 'local_x':
                lx = float(t.get('v'))
            elif t.get('k') == 'local_y':
                ly = float(t.get('v'))
        if lx is not None:
            nodes[n.get('id')] = (lx, ly)

    way_nodes = {w.get('id'): [nd.get('ref') for nd in w.findall('nd')]
                 for w in root.findall('way')}

    def centroid(node_refs):
        pts = [nodes[r] for r in node_refs if r in nodes]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    groups = []
    for r in root.findall('relation'):
        tags = {t.get('k'): t.get('v') for t in r.findall('tag')}
        if tags.get('subtype') != 'traffic_light':
            continue
        cents = []
        for m in r.findall('member'):
            if m.get('type') == 'way':
                c = centroid(way_nodes.get(m.get('ref'), []))
                if c:
                    cents.append(c)
        if not cents:
            continue
        x = sum(c[0] for c in cents) / len(cents)
        y = sum(c[1] for c in cents) / len(cents)
        groups.append((int(r.get('id')), x, y))
    return groups


def main():
    ap = argparse.ArgumentParser(description='Build carla_gt_bridge map_info.json from a Lanelet2 map.')
    ap.add_argument('--osm', required=True, help='path to the town lanelet2_map.osm')
    ap.add_argument('--out', required=True, help='output map_info.json path')
    args = ap.parse_args()

    groups = parse_traffic_lights(args.osm)
    if not groups:
        # Not an error: some Lanelet2 maps (e.g. CARLA Town02's community map) have no
        # traffic-light regulatory elements even though the sim has light actors. Autoware
        # then has nothing to obey, so traffic lights are simply unavailable for this town.
        print(f'WARNING: no traffic_light regulatory elements in {args.osm} -- '
              'traffic lights will be disabled for this town.')

    entries = [{'autoware_traffic_light': gid,
                'traffic_light_position': {'x': round(x, 3), 'y': round(y, 3)}}
               for gid, x, y in sorted(groups)]
    data = {'intersections': {'all': entries}}   # one flat group; carla_gt_bridge flattens anyway

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'wrote {args.out}: {len(entries)} traffic-light groups')


if __name__ == '__main__':
    main()
