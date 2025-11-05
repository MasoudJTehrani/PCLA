import os
from pathlib import Path
import yaml
import time

import cv2
import torch
import numpy as np
import carla

from dataset import generate_batch, split_large_BB
from lit_module import LitHFLM

from carla_garage.nav_planner import RoutePlanner

from rdp import rdp
import carla_garage.transfuser_utils as t_u
from carla_garage.lateral_controller import LateralPIDController
from carla_garage.longitudinal_controller import LongitudinalLinearRegressionController
from scipy.interpolate import PchipInterpolator

from leaderboard_codes.timer import GameTime

from leaderboard_codes import autonomous_agent1 as autonomous_agent
from leaderboard_codes.autonomous_agent1 import Track

from carla_garage.privileged_route_planner import PrivilegedRoutePlanner

from leaderboard_codes.carla_data_provider import CarlaDataProvider # privileged

from carla_garage.config import GlobalConfig

from leaderboard_codes.route_manipulation import downsample_route

from util.viz_batch import viz_batch

def get_entry_point():
    return 'PlanTAgent'

class PlanTAgent(autonomous_agent.AutonomousAgent):

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self.org_dense_route_gps = global_plan_gps
        self.org_dense_route_world_coord = global_plan_world_coord
        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):

        self.cleared_stop_sign = False
        self.step = 0
        self.initialized = False

        self.track = Track.MAP
        self.config = GlobalConfig()

        self.lat_pid = LateralPIDController(self.config)
        self.lon_ctrl = LongitudinalLinearRegressionController(self.config)

        with open(path_to_conf_file) as f:
            cfg = yaml.safe_load(f)

        self.visualize = cfg["visualize"]
        self.viz_path = os.path.join(cfg["viz_path"], time.strftime("%Y_%m_%d-%H:%M:%S"))
        os.makedirs(self.viz_path, exist_ok=True)
        LOAD_CKPT_PATH = cfg["checkpoint"]
        
        # Make the checkpoint path relative to the PCLA directory, not the current working directory
        if not os.path.isabs(LOAD_CKPT_PATH):
            # Get the PCLA root directory (where PCLA.py is located)
            pcla_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            LOAD_CKPT_PATH = os.path.join(pcla_root, LOAD_CKPT_PATH)

        print(f'Loading model from {LOAD_CKPT_PATH}')

        if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
            self.net = LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH) #, map_location="cpu")
        else:
            raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        self.net.eval()

        self.max_actor_distance = self.net.cfg_train.max_object_dist

        self.rdp_epsilon = 0.5 # epsilon for route shortening

    def _init(self, hd_map, vehicle):
        self._route_planner = RoutePlanner(7.5, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        # Get the hero vehicle and the CARLA world
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.world_map = self._world.get_map()

        distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(self._vehicle.get_location())
        # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
        # ego vehicle at the beginning.
        starts_with_parking_exit = distance_to_road > 2

        self._waypoint_planner = PrivilegedRoutePlanner(self.config)
        self._waypoint_planner.setup_route(self.org_dense_route_world_coord, self._world, self.world_map,
                                        starts_with_parking_exit, self._vehicle.get_location())
        self._waypoint_planner.save()

        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        self.initialized = True


    def sensors(self):
        result = [{
            "type": "sensor.other.imu",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "sensor_tick": 0.05,
            "id": "imu"
        }, 
        {
            "type": "sensor.speedometer",
            "reading_frequency": 20,
            "id": "speed"
        }]
        
        if self.visualize:
            result.append({
            'type': 'sensor.camera.rgb',
            'x': self.config.camera_pos[0],
            'y': self.config.camera_pos[1],
            'z': self.config.camera_pos[2],
            'roll': self.config.camera_rot_0[0],
            'pitch': self.config.camera_rot_0[1],
            'yaw': self.config.camera_rot_0[2],
            'width': 512,
            'height': 256,
            'fov': self.config.camera_fov,
            'id': 'rgb'
            })

        return result

    def tick(self, input_data):
        result = {}

        loc = self._vehicle.get_location()
        pos = np.array([loc.x, loc.y, loc.z])
        result['gps'] = pos[:2]

        speed = input_data['speed'][1]['speed']
        result["speed"] = speed

        compass = t_u.preprocess_compass(input_data['imu'][1][-1])
        result["yaw"] = t_u.normalize_angle(compass)

        if self.visualize:
            result["rgb"] = input_data["rgb"]

        waypoint_route = self._route_planner.run_step(pos)

        if len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]

        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], compass)

        result['target_point'] = tuple(ego_target_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None, vehicle=None):
        self.step += 1

        if not self.initialized:
            self._init(None, vehicle)

        tick_data = self.tick(input_data)

        boxes = self.get_bev_boxes()

        self.control = self._get_control(boxes, tick_data)

        inital_frames_delay = 40
        if self.step < inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        return self.control


    def _get_forward_speed(self, transform=None, velocity=None):
        """
                Calculate the forward speed of the vehicle based on its transform and velocity.

                Args:
                        transform (carla.Transform, optional): The transform of the vehicle. If not provided, it will be obtained from the vehicle.
                        velocity (carla.Vector3D, optional): The velocity of the vehicle. If not provided, it will be obtained from the vehicle.

                Returns:
                        float: The forward speed of the vehicle in m/s.
                """
        if velocity is None:
            velocity = self._vehicle.get_velocity()

        if transform is None:
            transform = self._vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array(
            [np.cos(pitch_rad) * np.cos(yaw_rad),
             np.cos(pitch_rad) * np.sin(yaw_rad),
             np.sin(pitch_rad)])

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return forward_speed

    def get_bev_boxes(self):
        # Ego infos
        ego_location = self._vehicle.get_location()
        ego_transform = self._vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_yaw =  ego_rotation.yaw
        ego_extent = self._vehicle.bounding_box.extent
        ego_speed = self._vehicle.get_velocity().length()

        # Bounding boxes
        boxes = []
        actors = self._world.get_actors()
        for cls_id, cls_filter in [(1, "*vehicle*"), (3, "*walker*"), (4, "*static*")]:
            cls_actors = actors.filter(cls_filter)
            for actor in cls_actors:
                if actor.id == self._vehicle.id or actor.get_location().distance(ego_location) > self.max_actor_distance:
                    continue

                actor_transform = actor.get_transform()
                actor_rotation = actor_transform.rotation
                actor_matrix = np.array(actor_transform.get_matrix())
                actor_extent = actor.bounding_box.extent

                relative_yaw = t_u.normalize_angle_degree(actor_rotation.yaw - ego_yaw)

                relative_pos = t_u.get_relative_transform(ego_matrix, actor_matrix)

                actor_velocity  = actor.get_velocity()
                actor_speed = self._get_forward_speed(transform=actor_transform, velocity=actor_velocity) # In m/s

                boxes.append([cls_id,
                            relative_pos[0], # X 
                            relative_pos[1], # Y 
                            relative_yaw,
                            actor_speed*3.6,
                            actor_extent.x,
                            actor_extent.y])

        # Route boxes
        self._waypoint_planner.load()
        waypoint_route, _, _, next_light_dist, next_light, next_stop_dist, next_stop, _ = self._waypoint_planner.run_step(np.array([ego_location.x, ego_location.y]))
        waypoint_route = waypoint_route[:250, :2] # only x,y and only 25m
        self._waypoint_planner.save()

        shortened_route = rdp(waypoint_route, epsilon=self.rdp_epsilon)

        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors/2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.rad2deg(np.arctan2(vectors[:,1], vectors[:,0]))

        route_boxes = []
        # We only want to keep max_NextRouteBBs boxes
        for i in range(min(len(midpoints), self.net.cfg_train.max_NextRouteBBs)):
            mid_matrix = np.eye(4)
            mid_matrix[:3, 3] = [*midpoints[i], ego_location.z]
            relative_pos = t_u.get_relative_transform(ego_matrix, mid_matrix)

            relative_yaw = t_u.normalize_angle_degree(angles[i] - ego_yaw)

            route_boxes.append([2,
                          relative_pos[0],
                          relative_pos[1],
                          relative_yaw,
                          i,
                          norms[i]/2,
                          ego_extent.y])

        # we split route segment slonger than 10m into multiple segments
        # improves generalization
        route_split = []
        for route in route_boxes:
            if route[5] > 5: # split at extent 5, so segments are no longer than 10m in total
                routes = split_large_BB(route, len(route_split))
                route_split.extend(routes)
            else:
                route_split.append(route)

        boxes.extend(route_split[:self.net.cfg_train.max_NextRouteBBs])

        # Calculate the accurate distance to the stop sign
        if next_stop is not None:
            distance_to_stop_sign = next_stop.get_transform().transform(next_stop.trigger_volume.location) \
                .distance(ego_location)
        else:
            distance_to_stop_sign = 999999999

        # Reset the stop sign cleared flag if we are farther than 10m away
        if distance_to_stop_sign > self.config.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
        # Set the stop sign cleared flag if we are closer than 3m and speed is low enough
        elif ego_speed < 0.1 and distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
            self.cleared_stop_sign = True


        # Stop signs
        if next_stop is not None and next_stop_dist < self.net.cfg_train.max_object_dist and not self.cleared_stop_sign:

            stop_trafo = next_stop.get_transform()
            trigger_bbox = next_stop.trigger_volume
            trigger_location = stop_trafo.location + trigger_bbox.location

            trigger_yaw = stop_trafo.rotation.yaw + trigger_bbox.rotation.yaw
            relative_yaw = t_u.normalize_angle_degree(trigger_yaw - ego_yaw)

            box_matrix = np.eye(4)
            box_matrix[:3, 3] = (trigger_location.x, trigger_location.y, trigger_location.z)
            relative_pos = t_u.get_relative_transform(ego_matrix, box_matrix)

            if np.linalg.norm(relative_pos[:2]) <= self.net.cfg_train.max_object_dist:
                boxes.append([6.0,
                            relative_pos[0], # X 
                            relative_pos[1], # Y 
                            relative_yaw,
                            0,
                            1.5,
                            1.5])

        # Traffic lights
        if next_light is not None and next_light_dist < self.net.cfg_train.max_object_dist and next_light.state in [carla.libcarla.TrafficLightState.Red, carla.libcarla.TrafficLightState.Yellow]:
            for wp in next_light.get_stop_waypoints():

                wp_transform = wp.transform
                wp_rotation = wp_transform.rotation
                wp_matrix = np.array(wp_transform.get_matrix())

                relative_yaw = t_u.normalize_angle_degree(wp_rotation.yaw - ego_yaw)

                relative_pos = t_u.get_relative_transform(ego_matrix, wp_matrix)

                boxes.append([7.0,
                            relative_pos[0], # X 
                            relative_pos[1], # Y 
                            relative_yaw,
                            0,
                            1.5,
                            1.5])

        return boxes

    def _get_control(self, boxes, input_data):

        gt_velocity = torch.FloatTensor([input_data['speed']])
        input_batch = self.get_input_batch(boxes, input_data)

        input_batch = {k: v.cuda() if v is not None else v for k, v in input_batch.items()}

        pred_wp = self.net(input_batch)[2]

        input_batch["waypoints"] = pred_wp.detach()
        
        if self.step%25==0 and self.visualize:
            img = viz_batch(input_batch, rgb=input_data["rgb"])
            cv2.imwrite(f"{self.viz_path}/{GameTime.get_frame()}.png", img)

        steer, throttle, brake = self.control_pid(pred_wp, gt_velocity)

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        return control

    # In: Waypoints NxD
    # Out: Waypoints NxD equally spaced 0.1 across D
    def interpolate_waypoints(self, waypoints):
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(waypoints-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists)) * 1e-4 # Prevents dists not being strictly increasing

        interp = PchipInterpolator(dists, waypoints, axis=0)

        x = np.arange(0.1, dists[-1], 0.1)

        interp_points = interp(x)

        # There is a possibility that all points are at 0, meaning there is no point distanced 0.1
        # In this case we output the last (assumed to be furthest) waypoint.
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]

        return interp_points

    def control_pid(self, waypoints, velocity):
        waypoints = waypoints.detach().cpu().squeeze().numpy()

        target_speed = np.linalg.norm(waypoints[-3] - waypoints[-1]) * 2.0 # Sampled at 4hz => 2 wps are 0.5s apart => dist * 2 = dist / second
        hazard_brake = target_speed < 0.05
        velocity = velocity.item()
        throttle, brake = self.lon_ctrl.get_throttle_and_brake(hazard_brake, target_speed, velocity)

        interp_wp = self.interpolate_waypoints(waypoints)

        # Decrease integral accumulation in standstill
        if velocity < 0.05 and target_speed < 0.05:
            interp_wp = np.array([[1,0], [2,0], [3,0], [4,0]])

        steer = self.lat_pid.step(interp_wp, velocity, np.array([0., 0.]), 0., False)

        return steer, throttle, brake

    def get_input_batch(self, boxes, input_data):

        sample = {'input': boxes, 
                  'output': [], 
                  'waypoints': [], 
                  'target_point': []}

        local_command_point = np.array([input_data['target_point'][0], input_data['target_point'][1]])
        sample['target_point'] = local_command_point

        batch = [sample]
        
        input_batch = generate_batch(batch)

        input_batch["y_objs"] = None # No forecasting at test time

        return input_batch

    def destroy(self, results=None): #TODO
        del self.net
        super().destroy()
