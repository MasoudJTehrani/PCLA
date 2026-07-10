# Copyright (c) 2025 Testing Automated group (TAU) at
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
"""Reference PCLA driver for the Autoware agent.

Preconditions (see memory `autoware-carla-phase1-proven`):
  * CARLA + zenoh_carla_bridge (Rust, WITHOUT its own carla_agent ego) + Autoware are already
    running (bring them up with `bash ~/start_stack.sh`).
  * Host user is in the `docker` group (else `ensure_docker_access()` re-execs under `sg docker`).

CARLA runs in SYNCHRONOUS mode, but the zenoh_carla_bridge is the tick master (it sets
synchronous_mode and calls world.tick() itself). So unlike sample.py we must NOT change the sync
setting and must NOT call world.tick(): PCLA is a passive participant, paced by the bridge via
world.wait_for_tick(). The ego is spawned with role_name "autoware_v1" so the bridge attaches to it.

The per-run bring-up helpers (docker access, bridge reset, stale-ego cleanup) live in
`pcla_functions/autoware_runtime.py` so your own driver scripts can reuse them.
"""

import time

import carla

from PCLA import PCLA
from pcla_functions import ensure_docker_access, reset_bridge, destroy_stale_ego

ROLE_NAME = "autoware_v1"          # must match "autoware_" + config vehicle_role
SPAWN_INDEX = 1                    # spawn point index (every town has more than one)


def main():
    ensure_docker_access()   # re-exec under `sg docker` if this shell lacks docker access
    reset_bridge()           # fresh bridge each run (a previous run's exit can crash it)
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)

    world = client.get_world()      # do NOT reload: keep Autoware's sensor stream alive
    print(f"Map loaded by the stack: {world.get_map().name} "
          f"(set the town via TOWN=... when you run start_stack.sh)")

    # The bridge is the sync-tick master; do NOT change sync mode or tick ourselves.
    assert world.get_settings().synchronous_mode, \
        "CARLA is not in synchronous mode -> the zenoh_carla_bridge is not running"

    world.wait_for_tick()
    destroy_stale_ego(world, ROLE_NAME)
    world.wait_for_tick()

    bp = world.get_blueprint_library().filter("model3")[0]
    bp.set_attribute("role_name", ROLE_NAME)   # bridge attaches by this
    spawn = world.get_map().get_spawn_points()[SPAWN_INDEX]
    vehicle = world.try_spawn_actor(bp, spawn)
    assert vehicle is not None, "failed to spawn ego"
    world.wait_for_tick()

    pcla = None
    try:
        # NOTE on routing: unlike the neural PCLA agents, Autoware is GOAL-based -- it does NOT
        # follow sample_route.xml's dense waypoints. It takes a single destination and plans its
        # own lane-level route to it on the HD map. sample_route.xml is still passed (PCLA parses
        # it), but the adapter only uses its LAST point, and only as a fallback: the destination is
        # picked as  goal_override > goal_ahead_m > this route endpoint  (all in config.yaml).
        # To choose the PATH (not just the endpoint), add ordered `checkpoints` in config.yaml.
        # See pcla_agents/autoware/README.md -> "Routing & goals".
        pcla = PCLA(ROLE_NAME, vehicle, "./sample_route.xml", client)
        print(f"\nSpawned ego (role={ROLE_NAME}); Autoware will localize, set goal, engage.\n"
              "Press Ctrl+C to stop.\n")
        spectator = world.get_spectator()
        step = 0
        while True:
            ego_action = pcla.get_action()
            if ego_action is not None:
                vehicle.apply_control(ego_action)  # redundant w/ bridge, kept for PCLA parity
            if step % 3 == 0:                       # chase-cam: keep the CARLA window on the ego
                tf = vehicle.get_transform()
                fwd = tf.get_forward_vector()
                spectator.set_transform(carla.Transform(
                    carla.Location(x=tf.location.x - 8 * fwd.x, y=tf.location.y - 8 * fwd.y,
                                   z=tf.location.z + 5),
                    carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)))
            world.wait_for_tick()
            step += 1
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        print(f"\nError: {type(e).__name__}: {e}\n")
        traceback.print_exc()
    finally:
        print("\nCleaning up.")
        if pcla is not None:
            pcla.cleanup()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
    print("Done.")
