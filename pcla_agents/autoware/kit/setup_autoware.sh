#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time setup for the PCLA Autoware agent. Brings the external stack from nothing to a
# state where `bash ~/start_stack.sh` works. Heavy steps (Docker image builds ~tens of GB,
# needing an NVIDIA GPU) are delegated to the upstream evshary/autoware_carla_launch, which
# documents them; this script wires them together and installs the PCLA kit on top.
#
# Prereqs: an NVIDIA GPU + driver, Docker + nvidia-container-toolkit, and a CARLA 0.9.16
# server (this repo assumes ~/carla_lite16/CarlaUE4.sh). See also kit/setup_autoware_host.sh
# for the apt/docker host packages. Run steps deliberately -- read the echoes.
# ---------------------------------------------------------------------------
set -e
AWROOT="${AWROOT:-$HOME/autoware_carla_launch}"
KIT="$(cd "$(dirname "$0")" && pwd)"
BRANCH=jazzy   # CARLA 0.9.16 / Autoware 1.8.0 / Zenoh 1.9.0 -- matches PCLA's CARLA

echo "== 1/5  clone autoware_carla_launch ($BRANCH) + zenoh_autoware_v2x =="
[ -d "$AWROOT" ] || git clone -b "$BRANCH" https://github.com/evshary/autoware_carla_launch "$AWROOT"
[ -d "$AWROOT/external/zenoh_autoware_v2x" ] || \
    git clone https://github.com/evshary/zenoh_autoware_v2x "$AWROOT/external/zenoh_autoware_v2x"

echo "== 2/5  build the two Docker images (LONG; needs --gpus all for the Autoware build) =="
echo "   Run these in $AWROOT (see upstream README for details):"
echo "     make prepare_bridge  && make build_bridge"
echo "     make prepare_autoware && make build_autoware"
read -r -p "   Press Enter once both images are built (or Ctrl-C to do it now and re-run)... " _

echo "== 3/5  create the aw_bridge + aw_autoware containers =="
echo "   Use the upstream container scripts (mount the repo, remap HOST_UID/GID, add --gpus all,"
echo "   and for RViz add X11: -e DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix):"
echo "     bash $AWROOT/container/run-bridge-docker.sh"
echo "     bash $AWROOT/container/run-autoware-docker.sh"
read -r -p "   Press Enter once 'docker ps' shows aw_bridge and aw_autoware Up... " _

echo "== 4/5  install the PCLA kit into the clone =="
bash "$KIT/install_kit.sh" "$AWROOT"

echo "== 5/5  fetch Town01's HD map (evshary's -- the one WITH traffic-light annotations) =="
( cd "$AWROOT" && bash script/setup/download_map.sh )
bash "$KIT/install_kit.sh" "$AWROOT"   # re-run so Town01's map_info lands at carla_maps/Town01/

echo
echo "Setup done. Start it:   bash ~/start_stack.sh        (headless: RENDER_OFFSCREEN=1 bash ~/start_stack.sh)"
echo "Then drive:             conda activate PCLA && cd ~/PCLA && python sample_autoware.py"
echo "Other towns:            bash ~/PCLA/pcla_agents/autoware/kit/prepare_town.sh Town03   (drive only; see README on traffic lights)"
