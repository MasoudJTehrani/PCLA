#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install the PCLA Autoware kit's runtime pieces into an autoware_carla_launch clone.
# Run this once after you have cloned evshary/autoware_carla_launch (jazzy branch), cloned
# zenoh_autoware_v2x into its external/, and built the two Docker images. Idempotent.
#
# Usage:  bash install_kit.sh [/path/to/autoware_carla_launch]   (default: $HOME/autoware_carla_launch)
# ---------------------------------------------------------------------------
set -e
AWROOT="${1:-$HOME/autoware_carla_launch}"
KIT="$(cd "$(dirname "$0")" && pwd)"
V2X="$AWROOT/external/zenoh_autoware_v2x"

[ -d "$AWROOT" ] || { echo "ERROR: autoware_carla_launch not found at $AWROOT (clone it first)."; exit 1; }
[ -d "$V2X" ]    || { echo "ERROR: clone zenoh_autoware_v2x into $AWROOT/external/ first (see README)."; exit 1; }

# 1) Traffic-light injector (runs inside aw_autoware via the zenoh_autoware_v2x uv env).
mkdir -p "$V2X/carla_gt_bridge"
cp "$KIT/carla_gt_bridge.py" "$V2X/carla_gt_bridge/main.py"

# 2) Autoware launch wrappers -- camera traffic-light recognition OFF (UNKNOWN in CARLA);
#    the injector provides ground-truth light states instead.
cp "$KIT/run_autoware_headless.sh" "$KIT/run_autoware_rviz.sh" "$AWROOT/"

# 3) One-shot stack bring-up.
cp "$KIT/start_stack.sh" "$HOME/start_stack.sh"

# 4) Make CARLA_MAP_NAME overridable in env.sh so TOWN switching works.
if grep -q 'export CARLA_MAP_NAME="Town01"' "$AWROOT/env.sh"; then
    sed -i 's|export CARLA_MAP_NAME="Town01"|export CARLA_MAP_NAME="${CARLA_MAP_NAME:-Town01}"|' "$AWROOT/env.sh"
fi

# 5) Town01's traffic-light table at the per-town location the adapter derives.
if [ -f "$V2X/map_info.json" ]; then
    mkdir -p "$V2X/carla_maps/Town01"
    cp "$V2X/map_info.json" "$V2X/carla_maps/Town01/map_info.json"
fi

echo "[install_kit] installed into $AWROOT."
echo "[install_kit] next:  bash ~/start_stack.sh   (then: python sample_autoware.py)"
