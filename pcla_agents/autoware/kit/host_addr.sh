#!/usr/bin/env bash
# Print the address of the host machine -- where CARLA (:2000) and aw_bridge's zenoh router
# (:7447) listen -- as seen from INSIDE the aw_autoware container.
#   * host networking (docker run --network host): 127.0.0.1
#   * Docker's bridge network (what upstream's container/run-autoware-docker.sh creates): the
#     container's default gateway, i.e. the host's docker0 address (usually 172.17.0.1). There,
#     127.0.0.1 is the container itself, so /clock and the sensors would never arrive.
# Override with AW_HOST_IP=<address> (e.g. CARLA + bridge on another machine).
if [ -n "${AW_HOST_IP:-}" ]; then
    echo "$AW_HOST_IP"
# A host-networked container sees the host's interfaces (docker0) and the zenoh router on loopback.
elif [ -e /sys/class/net/docker0 ] || timeout 1 bash -c '</dev/tcp/127.0.0.1/7447' 2>/dev/null; then
    echo 127.0.0.1
else
    gw=$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')
    echo "${gw:-127.0.0.1}"
fi
