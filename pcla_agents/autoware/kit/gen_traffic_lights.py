#!/usr/bin/env python
# Copyright (c) 2025 Testing Automated group (TAU) at
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
"""Annotate a LANE-ONLY Lanelet2 map with traffic-light regulatory elements from CARLA.

The community Lanelet2 maps (Town02-07, Town10HD) have lanes but NO traffic-light regulatory
elements, so Autoware has nothing to obey there. This tool adds them from the sim: for each
CARLA traffic light it reads the stop line (`traffic_light.get_stop_waypoints()`) and the light
pose, finds the lanelet the stop line sits on, and authors -- matching the structure Autoware's
Town01 map uses --

  * a `stop_line` way  (role ref_line: 2 nodes across the lane at the stop position)
  * a `traffic_light` way (role refers: 2 nodes at the light, elevated)
  * a `regulatory_element` relation (subtype=traffic_light) linking them
  * a `regulatory_element` member added to the lanelet relation

Nodes carry both local_x/local_y (Autoware `map` frame = CARLA with y negated) and lat/lon
(fit from the map's existing nodes, TransverseMercator origin 0,0,0). After running, `gen_map_info.py`
will find the new relations and carla_gt_bridge will inject their states.

Run with CARLA on the target town (bare CARLA; it only reads).
Usage:
    python gen_traffic_lights.py --town Town03 \
        --osm carla_map/Town03/lanelet2_map.osm --out carla_map/Town03/lanelet2_map.osm

STATUS: authoring verified against Town01's structure; live "does Autoware stop" validation
pending (needs a stable CARLA). See kit README notes.
"""
import argparse
import math
import xml.etree.ElementTree as ET

import carla


# ---- osm parsing -----------------------------------------------------------
def parse_osm(path):
    tree = ET.parse(path)
    root = tree.getroot()
    nodes = {}   # id -> dict(local_x, local_y, lat, lon)
    for n in root.findall('node'):
        d = {'lat': n.get('lat'), 'lon': n.get('lon')}
        for t in n.findall('tag'):
            d[t.get('k')] = t.get('v')
        nodes[n.get('id')] = d
    ways = {w.get('id'): [nd.get('ref') for nd in w.findall('nd')] for w in root.findall('way')}
    return tree, root, nodes, ways


def latlon_fit(nodes):
    """Least-squares ratios lon=a*local_x, lat=b*local_y from existing nodes (TM near origin)."""
    sx = sxx = sy = syy = 0.0
    ax = axx = ay = ayy = 0.0
    for d in nodes.values():
        if 'local_x' in d and d['lat'] is not None:
            x = float(d['local_x']); lo = float(d['lon'])
            y = float(d['local_y']); la = float(d['lat'])
            sx += x * lo; sxx += x * x
            sy += y * la; syy += y * y
            ax += x; axx += 1
    a = sx / sxx if sxx else 8.9868e-6   # lon per local_x
    b = sy / syy if syy else 9.0473e-6   # lat per local_y
    return a, b


def lanelet_centerlines(root, nodes, ways):
    """[(lanelet_id, [(x,y)...] centerline)] from left/right boundary ways (Autoware frame)."""
    out = []
    for r in root.findall('relation'):
        tags = {t.get('k'): t.get('v') for t in r.findall('tag')}
        if tags.get('type') != 'lanelet':
            continue
        left = right = None
        for m in r.findall('member'):
            if m.get('role') == 'left':
                left = ways.get(m.get('ref'))
            elif m.get('role') == 'right':
                right = ways.get(m.get('ref'))
        if not left or not right:
            continue

        def pts(w):
            return [(float(nodes[i]['local_x']), float(nodes[i]['local_y'])) for i in w if i in nodes]
        lp, rp = pts(left), pts(right)
        if not lp or not rp:
            continue
        m = min(len(lp), len(rp))
        center = [((lp[i][0] + rp[i][0]) / 2, (lp[i][1] + rp[i][1]) / 2) for i in range(m)]
        out.append((r.get('id'), r, center))
    return out


def nearest_lanelet(centerlines, sx, sy, hx, hy):
    """Lanelet whose centerline passes nearest (sx,sy) with direction aligned to (hx,hy)."""
    best = None
    for lid, rel, center in centerlines:
        for i in range(len(center) - 1):
            (x0, y0), (x1, y1) = center[i], center[i + 1]
            dx, dy = x1 - x0, y1 - y0
            seg = math.hypot(dx, dy)
            if seg < 1e-3:
                continue
            # distance from point to segment
            t = max(0.0, min(1.0, ((sx - x0) * dx + (sy - y0) * dy) / (seg * seg)))
            px, py = x0 + t * dx, y0 + t * dy
            d = math.hypot(sx - px, sy - py)
            align = (dx * hx + dy * hy) / seg
            if align < 0.5:
                continue
            if best is None or d < best[0]:
                best = (d, lid, rel)
    return best  # (dist, lanelet_id, relation) or None


# ---- XML authoring ---------------------------------------------------------
class Authoring:
    def __init__(self, root, nodes, a, b):
        self.root = root
        self.a, self.b = a, b
        self.next_id = max((int(e.get('id')) for e in root if e.get('id')), default=1000) + 1

    def _id(self):
        i = self.next_id
        self.next_id += 1
        return str(i)

    def node(self, x, y, ele):
        nid = self._id()
        n = ET.SubElement(self.root, 'node', {
            'id': nid, 'visible': 'true', 'version': '1',
            'lat': f'{y * self.b:.11f}', 'lon': f'{x * self.a:.11f}'})
        for k, v in (('local_x', f'{x:.4f}'), ('local_y', f'{y:.4f}'), ('ele', f'{ele:.1f}')):
            ET.SubElement(n, 'tag', {'k': k, 'v': v})
        return nid

    def way(self, node_ids, tags):
        wid = self._id()
        w = ET.SubElement(self.root, 'way', {'id': wid, 'visible': 'true', 'version': '1'})
        for nid in node_ids:
            ET.SubElement(w, 'nd', {'ref': nid})
        for k, v in tags.items():
            ET.SubElement(w, 'tag', {'k': k, 'v': v})
        return wid

    def reg_element(self, ref_line_way, refers_way):
        rid = self._id()
        r = ET.SubElement(self.root, 'relation', {'id': rid, 'visible': 'true', 'version': '1'})
        ET.SubElement(r, 'member', {'type': 'way', 'ref': ref_line_way, 'role': 'ref_line'})
        ET.SubElement(r, 'member', {'type': 'way', 'ref': refers_way, 'role': 'refers'})
        ET.SubElement(r, 'tag', {'k': 'type', 'v': 'regulatory_element'})
        ET.SubElement(r, 'tag', {'k': 'subtype', 'v': 'traffic_light'})
        return rid

    def link_to_lanelet(self, lanelet_rel, reg_id):
        ET.SubElement(lanelet_rel, 'member',
                      {'type': 'relation', 'ref': reg_id, 'role': 'regulatory_element'})


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Add traffic_light regulatory elements to a Lanelet2 map from CARLA.')
    ap.add_argument('--town', required=True)
    ap.add_argument('--osm', required=True, help='input lane-only lanelet2_map.osm')
    ap.add_argument('--out', required=True, help='output osm (may equal --osm)')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--max-dist', type=float, default=6.0, help='max stop-line -> lanelet distance (m)')
    args = ap.parse_args()

    tree, root, nodes, ways = parse_osm(args.osm)
    a, b = latlon_fit(nodes)
    centerlines = lanelet_centerlines(root, nodes, ways)
    auth = Authoring(root, nodes, a, b)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if args.town.lower() not in world.get_map().name.lower():
        world = client.load_world(args.town)
    lights = list(world.get_actors().filter('traffic.traffic_light'))

    added = 0
    for tl in lights:
        ltf = tl.get_transform()
        lx, ly, lz = ltf.location.x, -ltf.location.y, ltf.location.z  # -> Autoware frame
        for wp in tl.get_stop_waypoints():
            sx, sy = wp.transform.location.x, -wp.transform.location.y
            fwd = wp.transform.get_forward_vector()
            hx, hy = fwd.x, -fwd.y                                    # heading in Autoware frame
            hn = math.hypot(hx, hy) or 1.0
            hx, hy = hx / hn, hy / hn
            hit = nearest_lanelet(centerlines, sx, sy, hx, hy)
            if hit is None or hit[0] > args.max_dist:
                continue
            _, _lid, lanelet_rel = hit
            half = wp.lane_width / 2.0
            px, py = -hy, hx                                          # perpendicular (across the lane)
            sl = auth.way([auth.node(sx + px * half, sy + py * half, 0.0),
                           auth.node(sx - px * half, sy - py * half, 0.0)],
                          {'type': 'stop_line', 'subtype': 'solid'})
            tw = auth.way([auth.node(lx - px * 0.3, ly - py * 0.3, lz + 0.5),
                           auth.node(lx + px * 0.3, ly + py * 0.3, lz + 0.5)],
                          {'type': 'traffic_light', 'subtype': 'red_yellow_green', 'height': '0.5'})
            reg = auth.reg_element(sl, tw)
            auth.link_to_lanelet(lanelet_rel, reg)
            added += 1

    tree.write(args.out, encoding='UTF-8', xml_declaration=True)
    print(f'annotated {added} traffic-light stop lines into {args.out} '
          f'({len(lights)} CARLA lights; latlon fit a={a:.3e} b={b:.3e})')
    if added == 0:
        print('WARNING: nothing added -- check the osm has `type=lanelet` relations and the '
              'stop lines fall within --max-dist of a lanelet centerline.')


if __name__ == '__main__':
    main()
