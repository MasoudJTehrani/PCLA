# Copyright (c) 2025 Testing Automated group (TAU) at
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime helpers for the Autoware PCLA agent.

Unlike the other pcla_functions (route/waypoint/sensor utilities usable by any agent), these are
specific to the **Autoware** agent: they manage the zenoh_carla_bridge, the Docker control plane,
and the ego role_name the bridge attaches to. Import them into your own Autoware driver script:

    from pcla_functions import ensure_docker_access, reset_bridge, destroy_stale_ego

`sample_autoware.py` is the reference example; see `pcla_agents/autoware/README.md` for the design.
"""
import os
import subprocess
import sys
import time

import carla

DEFAULT_BRIDGE_CONTAINER = "aw_bridge"
DEFAULT_EGO_ROLE = "autoware_v1"

# One fresh zenoh_carla_bridge, detached, inside the bridge container.
_BRIDGE_CMD = (
    "cd ~/autoware_carla_launch && source env.sh && "
    "./external/zenoh_carla_bridge/target/release/zenoh_carla_bridge --mode ros2 "
    "--zenoh-listen tcp/0.0.0.0:7447 --zenoh-config ${ZENOH_CARLA_BRIDGE_CONFIG} "
    "--carla-address ${CARLA_SIMULATOR_IP} > /home/aw/autoware_carla_launch/run_bridge_only.log 2>&1"
)


def has_docker_access():
    """True if this process can reach the Docker daemon (needed for the Autoware control plane)."""
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def ensure_docker_access(script=None):
    """Guarantee Docker access, RE-EXECING the current script under `sg docker` if needed.

    The Autoware adapter drives Autoware via `docker exec`; if this shell isn't in the `docker`
    group those calls fail silently (the goal is never delivered -> the car won't move). If access
    is missing this replaces the process (os.execvp) with the same script run under `sg docker` --
    i.e. it does NOT return in that case. `script` defaults to the invoked script (sys.argv[0]).
    Use `has_docker_access()` instead if you only want the check without the re-exec.
    """
    if has_docker_access():
        return
    if os.environ.get("_PCLA_SG_REEXEC") == "1":
        print("WARNING: no docker access even under 'sg docker'. Autoware calls will fail.\n"
              "         Log out of your desktop and back in once to join the 'docker' group.")
        return
    os.environ["_PCLA_SG_REEXEC"] = "1"
    script = os.path.abspath(script or sys.argv[0])
    print("(this shell isn't in the 'docker' group; re-running under 'sg docker'...)")
    os.execvp("sg", ["sg", "docker", "-c", f"{sys.executable} {script}"])


def destroy_stale_ego(world, role_name=DEFAULT_EGO_ROLE):
    """Remove any leftover ego with this role_name (and its sensors) from a previous run."""
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") == role_name:
            for s in world.get_actors().filter("sensor.*"):
                if s.parent and s.parent.id == a.id:
                    s.destroy()
            a.destroy()


def reset_bridge(bridge_container=DEFAULT_BRIDGE_CONTAINER, host="localhost", port=2000):
    """Restart the zenoh_carla_bridge fresh for a run.

    The bridge crashes if it ever polls a destroyed ego, so a previous run's Ctrl+C can leave it
    dead (CARLA frozen). We kill it, remove stale `autoware_*` egos while nothing is attached (so
    the fresh bridge won't crash on their deletion), then start exactly one fresh bridge (which
    re-applies synchronous_mode and becomes the tick master).
    """
    subprocess.run(["docker", "exec", bridge_container, "pkill", "-9", "-f",
                    "zenoh_carla_bridge"], capture_output=True)
    time.sleep(1.5)
    try:
        client = carla.Client(host, port)
        client.set_timeout(20.0)
        world = client.get_world()
        st = world.get_settings()
        st.synchronous_mode = False
        world.apply_settings(st)
        try:
            world.tick()          # force-apply async even if CARLA was frozen in sync (no ticker)
        except Exception:
            pass
        for a in world.get_actors().filter("vehicle.*"):
            if (a.attributes.get("role_name") or "").startswith("autoware_"):
                for s in world.get_actors().filter("sensor.*"):
                    if s.parent and s.parent.id == a.id:
                        s.destroy()
                a.destroy()
    except Exception as e:
        print("reset_bridge: ego cleanup warning:", e)
    subprocess.run(["docker", "exec", "-d", "-u", "aw", bridge_container, "bash", "-c",
                    _BRIDGE_CMD], capture_output=True)
    for _ in range(20):            # wait until the fresh bridge takes over (sync mode on)
        time.sleep(1.0)
        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)
            if client.get_world().get_settings().synchronous_mode:
                print("bridge restarted fresh.")
                return
        except Exception:
            pass
    print("WARNING: bridge did not come up after reset_bridge; is CARLA running?")
