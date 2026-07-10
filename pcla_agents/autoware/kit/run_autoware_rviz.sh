#!/usr/bin/env bash
# Autoware launch WITH RViz (visual). Runs inside the aw_autoware container, which
# must have been started with X11 access (see the visual runbook: DISPLAY + /tmp/.X11-unix
# + NVIDIA_DRIVER_CAPABILITIES=all). Same as run_autoware_headless.sh but rviz:=true.
source /opt/ros/jazzy/setup.bash
source /opt/autoware/setup.bash
cd /home/aw/autoware_carla_launch
source env.sh
source install/setup.bash

LOG_PATH=autoware_log/rviz_$(date '+%Y-%m-%d_%H-%M-%S')
mkdir -p "${LOG_PATH}"
echo "logdir=${LOG_PATH}  DISPLAY=${DISPLAY}"

# Camera traffic-light recognition OFF (UNKNOWN in CARLA); the PCLA carla_gt_bridge
# injects ground-truth light state instead. Two publishers on the signals topic would fight.
ros2 launch autoware_carla_launch autoware_zenoh.launch.xml rviz:=true \
    use_traffic_light_recognition:=false \
    > "${LOG_PATH}/autoware.log" 2>&1 &
AW_PID=$!

"${AUTOWARE_CARLA_ROOT}/external/zenoh-plugin-ros2dds/target/release/zenoh-bridge-ros2dds" \
    -n "/${VEHICLE_NAME}" -d "${ROS_DOMAIN_ID}" -c "${ZENOH_BRIDGE_ROS2DDS_CONFIG}" \
    -e tcp/127.0.0.1:7447 -e tcp/127.0.0.1:7887 \
    > "${LOG_PATH}/zenoh_ros2dds.log" 2>&1 &
ZB_PID=$!

echo "autoware_pid=${AW_PID} zenoh_bridge_pid=${ZB_PID}"
wait
