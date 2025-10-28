# Copyright (c) 2025 Testing Automated group (TAU) at 
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0

import importlib
import os
import sys

# Ensure we can import pcla_functions regardless of where this script is called from
pcla_dir = os.path.dirname(os.path.abspath(__file__))
if pcla_dir not in sys.path:
    sys.path.insert(0, pcla_dir)

import carla
import traceback
# give_path, setup_sensor_attributes, location_to_waypoint, route_maker
from pcla_functions import give_path, setup_sensor_attributes, location_to_waypoint, route_maker
from leaderboard_codes.watchdog import Watchdog
from leaderboard_codes.timer import GameTime
from leaderboard_codes.route_indexer import RouteIndexer
from leaderboard_codes.route_manipulation import interpolate_trajectory
from leaderboard_codes.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader

class PCLA():
    def __init__(self, agent, vehicle, route, client):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.client = None
        self.world = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.agent_instance = None
        self.routePath = None
        self._watchdog = None
        self.set(agent, vehicle, route, client)
    
    def set(self, agent, vehicle, route, client):
        self.client = client
        self.world = client.get_world()
        self.vehicle = vehicle
        self.routePath = route
        self._watchdog = Watchdog(260) # TODO: Increase timeout if needed for large models
        self.setup_agent(agent)
        self.setup_route()
        self.setup_sensors()

    def setup_agent(self, agent):
        GameTime.restart()
        self._watchdog.start()
        self.agentPath, self.configPath = give_path(agent, self.current_dir, self.routePath)

        module_name = os.path.basename(self.agentPath).split('.')[0]
        sys.path.insert(0, os.path.dirname(self.agentPath))
        module_agent = importlib.import_module(module_name)
        
        agent_class_name = getattr(module_agent, 'get_entry_point')()
        self.agent_instance = getattr(module_agent, agent_class_name)(self.configPath)

        self._watchdog.stop()

    def setup_route(self):
        scenarios = os.path.join(self.current_dir, "leaderboard_codes/no_scenarios.json")
        route_indexer = RouteIndexer(self.routePath, scenarios, 1)
        config = route_indexer.next()
        
        gps_route, route = interpolate_trajectory(self.world, config.trajectory)

        self.agent_instance.set_global_plan(gps_route, route)

    def setup_sensors(self):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        """
        bp_library = self.world.get_blueprint_library()
        for sensor_spec in self.agent_instance.sensors():
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = OpenDriveMapReader(self.vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = 1/20
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(self.vehicle, frame_rate)
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                bp_setup = setup_sensor_attributes(bp, sensor_spec)
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z'])
                if sensor_spec['type'].startswith('sensor.other.gnss'):
                    sensor_rotation = carla.Rotation()
                else:
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'], roll=sensor_spec['roll'], yaw=sensor_spec['yaw'])

                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self.world.spawn_actor(bp_setup, sensor_transform, self.vehicle)
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.agent_instance.sensor_interface))

        # Tick once to spawn the sensors
        self.world.tick()
            
    def get_action(self):
        snapshot = self.world.get_snapshot()
        if snapshot:
            timestamp = snapshot.timestamp
        if timestamp:
            GameTime.on_carla_tick(timestamp)
            return self.agent_instance(vehicle = self.vehicle)
    
    def cleanup(self):
        """
        Remove and destroy all actors
        """
        if self._watchdog:
            self._watchdog.stop()

        # Cleanup the agent BEFORE destroying the vehicle
        try:
            if self.agent_instance:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        self.vehicle.destroy()
        self.current_dir = None
        self.client = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.routePath = None

        # Make sure no sensors are left streaming
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            if sensor.is_listening():
                sensor.stop()
            sensor.destroy()

        self.world = None
        