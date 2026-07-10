#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot clean startup for the PCLA <-> Autoware stack on vortex.
# Kills anything already running, then brings up EXACTLY ONE of each:
#   CARLA (Town01)  ->  zenoh_carla_bridge  ->  Autoware
# When it says READY, run the agent:  conda activate PCLA && cd ~/PCLA && python sample_autoware.py
#
# Usage:
#   bash ~/start_stack.sh                 # windowed CARLA, Epic quality (default; heavier GPU)
#   CARLA_QUALITY=Low bash ~/start_stack.sh    # lighter GPU load (lower fidelity)
#   RENDER_OFFSCREEN=1 bash ~/start_stack.sh   # headless CARLA (no window, most reliable)
#   CARLA_EXTRA_ARGS=-vulkan bash ~/start_stack.sh   # pass extra CarlaUE4 flags (e.g. -vulkan if the
#                                                    # default renderer segfaults / shows a black window)
# ---------------------------------------------------------------------------
# Paths (override via env for non-default installs):
CARLA_DIR="${CARLA_DIR:-$HOME/carla_lite16}"     # dir containing CarlaUE4.sh
AWROOT="${AWROOT:-$HOME/autoware_carla_launch}"  # the evshary/autoware_carla_launch clone
PCLA_PY="${PCLA_PY:-python}"                      # python with the CARLA 0.9.16 client (activate the PCLA env)
CARLA_QUALITY="${CARLA_QUALITY:-Epic}"
# TOWN : which CARLA map to load. Its HD map must exist at carla_map/$TOWN/ (lanelet2 + pcd +
#        projector). Use kit/prepare_town.sh to fetch/generate a town's map first.
TOWN="${TOWN:-Town01}"
# RVIZ=1 : run CARLA HEADLESS (no window jitter -> reliable) and show Autoware's RViz
#          instead (car + HD map + LiDAR + planned path). Needs aw_autoware created
#          with X11 + `xhost +local:`.
RVIZ="${RVIZ:-0}"
[ "$RVIZ" = "1" ] && RENDER_OFFSCREEN=1
RENDER=""
[ "${RENDER_OFFSCREEN:-0}" = "1" ] && RENDER="-RenderOffScreen"

# Use sudo for docker only if the docker group isn't active in this shell.
if docker ps >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
echo "[start_stack] docker command: $DOCKER"

# --- sanity: containers must exist ---
if ! $DOCKER ps -a --format '{{.Names}}' | grep -q '^aw_bridge$' \
   || ! $DOCKER ps -a --format '{{.Names}}' | grep -q '^aw_autoware$'; then
    echo "[start_stack] ERROR: aw_bridge / aw_autoware containers not found. Ask Claude to recreate them."
    exit 1
fi

echo "[start_stack] 1/6  killing anything already running..."
pkill -INT -f sample_autoware.py 2>/dev/null
pkill -9   -f CarlaUE4           2>/dev/null
sleep 2
# Restarting the containers wipes ALL processes inside (kills any duplicate bridge/Autoware).
$DOCKER restart aw_bridge aw_autoware >/dev/null
sleep 3

echo "[start_stack] 2/6  starting CARLA (quality=$CARLA_QUALITY ${RENDER:-windowed})..."
( cd "$CARLA_DIR" && nohup ./CarlaUE4.sh -quality-level=$CARLA_QUALITY -world-port=2000 -prefernvidia $RENDER ${CARLA_EXTRA_ARGS:-} \
    > /tmp/carla_stack.log 2>&1 & )
for i in $(seq 1 45); do ss -ltn 2>/dev/null | grep -q ':2000' && break; sleep 2; done
if ! ss -ltn 2>/dev/null | grep -q ':2000'; then
    echo "[start_stack] ERROR: CARLA didn't open port 2000 (see /tmp/carla_stack.log). If the window is black, add -vulkan."
    exit 1
fi
sleep 5

if [ ! -f "$AWROOT/carla_map/$TOWN/pointcloud_map.pcd" ] || [ ! -f "$AWROOT/carla_map/$TOWN/lanelet2_map.osm" ]; then
    echo "[start_stack] ERROR: HD map for $TOWN not found at carla_map/$TOWN/ (need lanelet2_map.osm + pointcloud_map.pcd)."
    echo "               Prepare it first:  bash ~/PCLA/pcla_agents/autoware/kit/prepare_town.sh $TOWN"
    exit 1
fi
echo "[start_stack] 3/6  loading $TOWN..."
$PCLA_PY -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(60); c.load_world('$TOWN')" \
    || { echo "[start_stack] ERROR loading $TOWN"; exit 1; }

echo "[start_stack] 4/6  starting the bridge (ONE instance, detached)..."
$DOCKER exec -d -u aw aw_bridge bash -c 'cd ~/autoware_carla_launch && source env.sh && ./external/zenoh_carla_bridge/target/release/zenoh_carla_bridge --mode ros2 --zenoh-listen tcp/0.0.0.0:7447 --zenoh-config ${ZENOH_CARLA_BRIDGE_CONFIG} --carla-address ${CARLA_SIMULATOR_IP} > /home/aw/autoware_carla_launch/run_bridge_only.log 2>&1'
for i in $(seq 1 20); do ss -ltn 2>/dev/null | grep -q ':7447' && break; sleep 1; done
if ! ss -ltn 2>/dev/null | grep -q ':7447'; then
    echo "[start_stack] ERROR: bridge didn't start (see the container's run_bridge_only.log). Is CARLA up?"
    exit 1
fi

echo "[start_stack] 5/6  starting Autoware (ONE instance, detached)..."
if [ "$RVIZ" = "1" ]; then
    echo "           (RViz mode - a window will open on your screen)"
    $DOCKER exec -d -u aw -e CARLA_MAP_NAME=$TOWN aw_autoware bash /home/aw/autoware_carla_launch/run_autoware_rviz.sh
else
    $DOCKER exec -d -u aw -e CARLA_MAP_NAME=$TOWN aw_autoware bash /home/aw/autoware_carla_launch/run_autoware_headless.sh
fi

echo "[start_stack] 6/6  waiting for Autoware to come up (~45s) and verifying single instance..."
$DOCKER exec -u aw aw_autoware bash -c '
source /opt/ros/jazzy/setup.bash >/dev/null 2>&1; source /opt/autoware/setup.bash >/dev/null 2>&1
cd ~/autoware_carla_launch && source env.sh >/dev/null 2>&1 && source install/setup.bash >/dev/null 2>&1
n=0; for i in $(seq 1 40); do n=$(ros2 node list 2>/dev/null | wc -l); [ "$n" -gt 60 ] && break; sleep 5; done
dups=$(ros2 node list 2>/dev/null | sort | uniq -d | tr "\n" " ")
echo "    nodes: $n"
echo "    duplicate node names (MUST be empty): [$dups]"
echo -n "    clock rate (want ~20 Hz): "; timeout 8 ros2 topic hz /clock 2>/dev/null | grep -m1 average || echo none
'

echo
echo "==================================================================="
echo "  STACK READY."
echo "  Now run the agent in another terminal:"
echo "      conda activate PCLA && cd ~/PCLA && python sample_autoware.py"
echo
echo "  Do NOT start the bridge or Autoware yourself - this script already"
echo "  started exactly one of each. To reset, just re-run this script."
echo "  To stop everything:  pkill -9 -f CarlaUE4 ; $DOCKER stop aw_bridge aw_autoware"
echo "==================================================================="
