# Copyright (c) 2025 Testing Automated group (TAU) at the università della svizzera 
# italiana (USI) Switzerland
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import importlib
import os
import sys
# Get the directory of the current file (PCLA.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the directory containing leaderboardcodes to the Python path
sys.path.append(current_dir)

import sys
import carla
import traceback
import json
from leaderboardcodes.watchdog import Watchdog
from leaderboardcodes.timer import GameTime
from leaderboardcodes.route_indexer import RouteIndexer
from leaderboardcodes.route_manipulation import interpolate_trajectory
from leaderboardcodes.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader
from leaderboardcodes.global_route_planner import GlobalRoutePlanner
from leaderboardcodes.global_route_planner_dao import GlobalRoutePlannerDAO

def print_guide():
    print("""
          -------------------------------
          The guide to this framework available at
          https://github.com/MasoudJTehrani/PCLA

          env variables (remember to unset these variables before using another agent):
            tfpp_l6_#:
                export UNCERTAINTY_THRESHOLD=033
            tfpp_lav_#:
                export STOP_CONTROL=1
            tfpp_aim_#:
                export DIRECT=0
            tfpp_wp_#:
                export DIRECT=0
            if_if:
                export ROUTES=path_to_route.xml
          -------------------------------
          """)
def location_to_waypoint(client, starting_location, ending_location, distance = 2, draw=False):
    # This function is used to generate waypoints between two locations
    world = client.get_world()
    amap = world.get_map()
    sampling_resolution = distance # the distance between the waypoints
    dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    w1 = grp.trace_route(starting_location, ending_location)
    # draw the route on the carla simulator
    if(draw):
        i = 0
        for w in w1:
            if i % 10 == 0:
                world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0), life_time=60.0,
                persistent_lines=True)
            else:
                world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                color = carla.Color(r=0, g=0, b=255), life_time=60.0,
                persistent_lines=True)
                i += 1
    
    return [wp[0] for wp in w1]

def route_maker(waypoints, savePath="route.xml"):
    # This function gets a list of carla waypoints and convert it into leaderboard route
    # This way you can use the route in PCLA
    from xml.dom import minidom 

    if(len(waypoints) == 1):
        print("Please provide more that 1 waypoint")
        return

    root = minidom.Document()
    root.toxml(encoding="utf-8")
  
    xml = root.createElement('route')
    xml.setAttribute('id', "_")
    xml.setAttribute( 'town', "_")
    root.appendChild(xml)
    
    for wp in waypoints:
        tf = wp.transform
        productChild = root.createElement('waypoint')
        productChild.setAttribute('pitch', str(tf.rotation.pitch))
        productChild.setAttribute('roll', str(tf.rotation.roll))
        productChild.setAttribute('x', str(tf.location.x))
        productChild.setAttribute('y', str(tf.location.y))
        productChild.setAttribute('yaw', str(tf.rotation.yaw))
        productChild.setAttribute('z', str(tf.location.z))
        xml.appendChild(productChild)
    
    xml_str = root.toprettyxml(indent ="\t")
    
  
    with open(savePath, "w") as f: 
        f.write(xml_str)

    return

def give_path(name):
    nameArray = name.split("_") # Split the name by _
    if(nameArray[0] != "tfpp"): # Handling the numbers from the tfpp agent name
        nameArray.append("")
   
    # open json file
    with open(current_dir + "/agents.json", 'r') as file:
        agentsFile = json.load(file)

        # check environment variables
        envs = agentsFile[nameArray[0]][nameArray[1]]["envs"]
        for var in envs:
            if(var not in os.environ):
                raise Exception(f"Please export the related environment variables\
                                and unset previous variables\nRequired variables: {envs}")
            
        # get agent and it's config path
        try:
            agent = agentsFile[nameArray[0]][nameArray[1]]["agent"]
            config = agentsFile[nameArray[0]][nameArray[1]]["config"] + nameArray[2] # Handling the numbers of agent names for tfpp
        except:
            print("couldn't find your model")
            print_guide()

    return current_dir + agent, current_dir + config


class PCLA():
    def __init__(self, agent, vehicle, route, client):
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
        self._watchdog = Watchdog(160) # TODO: Change watchdog timeout based on your internet speed for loading the agents
        self.setup_agent(agent)
        self.setup_route()
        self.setup_sensors()

    def setup_agent(self, agent):
        GameTime.restart()
        self._watchdog.start()
        self.agentPath, self.configPath = give_path(agent)
        
        module_name = os.path.basename(self.agentPath).split('.')[0]
        sys.path.insert(0, os.path.dirname(self.agentPath))
        module_agent = importlib.import_module(module_name)
        
        agent_class_name = getattr(module_agent, 'get_entry_point')()
        self.agent_instance = getattr(module_agent, agent_class_name)(self.configPath)

        self._watchdog.stop()

    def setup_route(self):
        
        scenarios = current_dir + "/leaderboardcodes/no_scenarios.json"
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
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(10))
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(600000))
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    bp.set_attribute('dropoff_general_rate', str(0.45))
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])

                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    bp.set_attribute('noise_alt_stddev', str(0.000005))
                    bp.set_attribute('noise_lat_stddev', str(0.000005))
                    bp.set_attribute('noise_lon_stddev', str(0.000005))
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self.world.spawn_actor(bp, sensor_transform, self.vehicle)
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
            return(self.agent_instance())
    
    def cleanup(self):
        """
        Remove and destroy all actors
        """
        # Cleanup the vehicle
        self.vehicle.destroy()
        self.client = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.routePath = None

        if self._watchdog:
            self._watchdog.stop()

        # Cleanup the agent
        try:
            if self.agent_instance:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        # Make sure no sensors are left streaming
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            if sensor.is_listening():
                sensor.stop()
            sensor.destroy()
        

        self.world = None
            