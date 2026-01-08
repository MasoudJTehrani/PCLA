# Copyright (c) 2025 Testing Automated group (TAU) at 
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0

import importlib
import os
import sys

# Ensure imports work regardless of caller's working directory
pcla_dir = os.path.dirname(os.path.abspath(__file__))
if pcla_dir not in sys.path:
    sys.path.insert(0, pcla_dir)

# Add lmdrive's custom timm (vision_encoder) before anything that might import timm
lmdrive_vision_encoder = os.path.join(pcla_dir, 'pcla_agents', 'lmdrive', 'vision_encoder')
if os.path.exists(lmdrive_vision_encoder) and lmdrive_vision_encoder not in sys.path:
    sys.path.insert(0, lmdrive_vision_encoder)

import carla
import traceback
from pcla_functions import give_path, setup_sensor_attributes, location_to_waypoint, route_maker
from leaderboard_codes.watchdog import Watchdog
from leaderboard_codes.timer import GameTime
from leaderboard_codes.route_indexer import RouteIndexer
from leaderboard_codes.carla_data_provider import CarlaDataProvider
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
        # Reload the agent freshly each time to avoid stale imports across agents.
        # Use file-based loading to prevent cross-talk between agents sharing module names (e.g., model.py).
        module_name = os.path.basename(self.agentPath).split('.')[0]
        module_dir = os.path.dirname(self.agentPath)
        module_key = f"pcla_dynamic_agent.{module_name}"

        # Prepend the agent directory so its relative imports resolve to local files.
        # Save/restore sys.path to isolate dependencies per agent.
        original_sys_path = list(sys.path)
        if module_dir in sys.path:
            sys.path.remove(module_dir)
        sys.path.insert(0, module_dir)

        # Drop any previously loaded modules with common local names to avoid cross-agent contamination.
        # Also clear 'models' package and all its submodules (models.bev_planner, models.lidar, etc.)
        for key in list(sys.modules.keys()):
            if key in (module_key, 'model', 'models', 'nav_planner', 'config', 'transfuser', 'utils', 'planner', 
                    'controller', 'map_agent', 'base_agent', 'bev_planner', 'waypointer') or key.startswith('models.'):
                del sys.modules[key]

        try:
            spec = importlib.util.spec_from_file_location(module_key, self.agentPath)
            module_agent = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module_agent
            spec.loader.exec_module(module_agent)
        finally:
            # Restore sys.path
            sys.path = original_sys_path
        
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
        """Attach sensors defined by the agent to the ego-vehicle."""
        bp_library = self.world.get_blueprint_library()
        for sensor_spec in self.agent_instance.sensors():
            # Pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                sensor = OpenDriveMapReader(self.vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = 1/20
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(self.vehicle, frame_rate)
            else:
                # World sensors (spawned actors)
                bp = bp_library.find(str(sensor_spec['type']))
                bp_setup = setup_sensor_attributes(bp, sensor_spec)
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z'])
                if sensor_spec['type'].startswith('sensor.other.gnss'):
                    sensor_rotation = carla.Rotation()
                else:
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'], roll=sensor_spec['roll'], yaw=sensor_spec['yaw'])

                # Create sensor actor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self.world.spawn_actor(bp_setup, sensor_transform, self.vehicle)
            # Register callback
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.agent_instance.sensor_interface))

        # Ensure sensors are created in the world
        self.world.tick()
        CarlaDataProvider.register_actor(self.vehicle)
            
    def get_action(self):
        snapshot = self.world.get_snapshot()
        if snapshot:
            timestamp = snapshot.timestamp
        if timestamp:
            GameTime.on_carla_tick(timestamp)
            return self.agent_instance(vehicle = self.vehicle)
    
    def cleanup(self):
        """Remove and destroy all actors."""

        if self._watchdog:
            self._watchdog.stop()

        # Cleanup the agent first so it can stop any internal threads and sensors
        try:
            if self.agent_instance is not None:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        # Stop and destroy any remaining sensors
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            if sensor.is_listening():
                sensor.stop()
            sensor.destroy()

        # Destroy the vehicle after sensors are cleaned up
        try:
            if self.vehicle is not None and self.vehicle.is_alive:
                self.vehicle.destroy()
        except RuntimeError:
            pass

        self.current_dir = None
        self.client = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.routePath = None
        self.world = None
        
        CarlaDataProvider.cleanup()
        