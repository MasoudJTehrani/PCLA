#!/usr/bin/env bash
# Headless Autoware launch for Phase-1 proof (no RViz).
# Runs inside the aw_autoware container (repo mounted at /home/aw/autoware_carla_launch).
source /opt/ros/jazzy/setup.bash
source /opt/autoware/setup.bash
cd /home/aw/autoware_carla_launch
source env.sh
source install/setup.bash

LOG_PATH=autoware_log/headless_$(date '+%Y-%m-%d_%H-%M-%S')
mkdir -p "${LOG_PATH}"
echo "logdir=${LOG_PATH}"

# 1) Autoware stack (no rviz). Camera-based traffic-light recognition is OFF: it returns
#    UNKNOWN on CARLA's rendered lights. The PCLA carla_gt_bridge injects CARLA's
#    ground-truth light state into /perception/traffic_light_recognition/traffic_signals
#    instead (the planner still obeys that topic). Two publishers would fight, so keep this false.
ros2 launch autoware_carla_launch autoware_zenoh.launch.xml rviz:=false \
    use_traffic_light_recognition:=false \
    > "${LOG_PATH}/autoware.log" 2>&1 &
AW_PID=$!

# 2) Zenoh <-> ROS2 DDS bridge (connects Autoware DDS to the Carla-side zenoh router on 7447)
"${AUTOWARE_CARLA_ROOT}/external/zenoh-plugin-ros2dds/target/release/zenoh-bridge-ros2dds" \
    -n "/${VEHICLE_NAME}" -d "${ROS_DOMAIN_ID}" -c "${ZENOH_BRIDGE_ROS2DDS_CONFIG}" \
    -e tcp/127.0.0.1:7447 -e tcp/127.0.0.1:7887 \
    > "${LOG_PATH}/zenoh_ros2dds.log" 2>&1 &
ZB_PID=$!

echo "autoware_pid=${AW_PID} zenoh_bridge_pid=${ZB_PID}"
wait
