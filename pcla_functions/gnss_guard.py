"""GNSS convention guard for CARLA 0.9.16.

CARLA 0.9.16 flipped the sign of the GNSS *latitude* relative to the convention
used by the route (``_location_to_gps``, which PCLA feeds to agents via
``set_global_plan``). Agents localize by converting the live GNSS reading into a
position and matching it against those route waypoints, so a flipped latitude
mirrors the ego: the route appears on the wrong side and the agent either veers
off-route immediately (TransFuser++) or gets a nonsense target point, brakes and
stalls (ThinkTwice).

The fix is to detect the sign convention **once** per route by trying all four
lat/lon sign combinations and keeping the one whose converted position best
matches a known reference. The wrong latitude sign is a gross mirror (it roughly
negates a coordinate that is hundreds of metres from the origin), so the correct
combination wins by a wide margin even when the reference is only approximate
(e.g. the ego spawns a few metres from the first route waypoint).

This resolves to the identity ``(+1, +1)`` on CARLA 0.9.15, so healthy setups are
unchanged; it only corrects 0.9.16. Only the *live GNSS* is touched — the route
conversion is left alone.

Each agent family converts GNSS to a position differently, so callers pass their
own ``to_position`` callable and a ``reference`` expressed in that same frame:

* ``(gps - mean) * scale``      -> reference = first route waypoint
* ``convert_gps_to_carla(gps)`` -> reference = true ego pose, else route start
* ``Waypointer.latlon_to_xy``   -> reference = first route waypoint
"""

import numpy as np

__all__ = ['detect_gps_sign', 'GnssSignGuard']

_SIGNS = (1.0, -1.0)


def detect_gps_sign(raw_gps, to_position, reference):
    """Return the ``(lat_sign, lon_sign)`` best matching ``reference``.

    Args:
        raw_gps: raw GNSS reading; ``(lat, lon)`` or ``(lat, lon, alt)``.
        to_position: callable taking ``(lat, lon)`` and returning a 2D position
            in the same frame as ``reference``.
        reference: 2D ground-truth-ish position used to score each combination.

    Returns:
        ``(lat_sign, lon_sign)``, each ``+1.0`` or ``-1.0``.
    """
    ref = np.asarray(reference, dtype=np.float64)
    best = None
    for lat_sign in _SIGNS:
        for lon_sign in _SIGNS:
            pos = np.asarray(
                to_position((float(raw_gps[0]) * lat_sign, float(raw_gps[1]) * lon_sign)),
                dtype=np.float64,
            )
            err = float((pos[0] - ref[0]) ** 2 + (pos[1] - ref[1]) ** 2)
            if best is None or err < best[0]:
                best = (err, lat_sign, lon_sign)
    return best[1], best[2]


class GnssSignGuard:
    """Stateful one-shot wrapper around :func:`detect_gps_sign`.

    Stays uncalibrated (and therefore identity) until a reference is available,
    so it is safe to call from the first tick before the route planner is
    populated. ``agent_name`` is only used to label the one-time log line.
    """

    def __init__(self, agent_name=''):
        self.lat_sign = 1.0
        self.lon_sign = 1.0
        self.calibrated = False
        self._agent_name = agent_name

    def reset(self):
        """Force re-detection (call when a new route/global plan is set)."""
        self.lat_sign = 1.0
        self.lon_sign = 1.0
        self.calibrated = False

    def calibrate(self, raw_gps, to_position, reference):
        """Detect the sign convention once. No-op once calibrated."""
        if self.calibrated or reference is None:
            return
        self.lat_sign, self.lon_sign = detect_gps_sign(raw_gps, to_position, reference)
        self.calibrated = True
        if self.lat_sign != 1.0 or self.lon_sign != 1.0:
            tag = f'[{self._agent_name}] ' if self._agent_name else ''
            print(f'{tag}GNSS sign guard: lat*={self.lat_sign:+.0f} lon*={self.lon_sign:+.0f} '
                  f'(CARLA 0.9.16 flip corrected)')

    def apply(self, raw_gps):
        """Return ``raw_gps`` with the detected signs applied.

        Preserves the input length, so a ``(lat, lon, alt)`` triple stays a
        triple (altitude is passed through untouched).
        """
        gps = np.array(raw_gps, dtype=np.float64)
        gps[0] *= self.lat_sign
        gps[1] *= self.lon_sign
        return gps
