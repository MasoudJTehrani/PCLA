import os
import re
import json
import yaml
from pathlib import Path
from datetime import datetime

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator

import carla
from leaderboard_codes.timer import GameTime

from carla_garage.data_agent import DataAgent
from dataset import generate_batch 
from lit_module import LitHFLM
from carla_garage import transfuser_utils as t_u
from carla_garage.lateral_controller import LateralPIDController
from util.viz_batch import viz_batch
from carla_garage.birds_eye_view.chauffeurnet import ObsManager
from carla_garage.longitudinal_controller import LongitudinalLinearRegressionController
from plant_variables import PlanTVariables
from util.static_extents import STATIC_EXTENTS, CAR_EXTENTS

def get_entry_point():
    return 'PlanTAgent'


def rad2deg(theta):
    return t_u.normalize_angle_degree(np.rad2deg(theta))

class PlanTAgent(DataAgent):
    # Defaults for robustness if sensors() is called before full setup in custom loaders.
    step = 0
    visualize_plant = False
    use_rgb = True
    img_path = ""
    input_bev = False
    initialized = False

    def setup(self,
              path_to_conf_file,
              route_index=None,
              cfg=None,
              exec_or_inter=None,
              traffic_manager=None):
        
        self.initialized = False
        self.control_history = [(0,0,0) for _ in range(25)]

        self.img_path = os.environ["PLANT_VIZ"]
        self.visualize_plant = len(self.img_path) > 0
        if self.visualize_plant:
            self.img_path = os.path.join(self.img_path, datetime.now().strftime("%H:%M:%S_%d-%m-%Y"))
            os.makedirs(self.img_path, exist_ok=True)
        
        self.use_rgb = True # eval_config["viz_img"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.route_index = route_index
        print("Route index:", self.route_index)

        super().setup(path_to_conf_file,
                  route_index=route_index,
                  cfg=cfg,
                  exec_or_inter=exec_or_inter,
                  traffic_manager=traffic_manager)

        LOAD_CKPT_PATH = os.environ["PLANT_CHECKPOINT"]

        self.cfg_net = torch.load(LOAD_CKPT_PATH, map_location="cpu", weights_only=False)["hyper_parameters"]["cfg"]
        self.input_bev = self.cfg_net["model"]["training"].get("input_bev", False)
        self.input_static_cars = self.cfg_net["model"]["training"].get("input_static_cars", False)

        self.input_range = self.cfg_net["model"]["training"].get("range", False)
        self.input_range_factor_front = self.cfg_net["model"]["training"].get("range_factor_front", False)

        print(f"BEV: {self.input_bev}, Static: {self.input_static_cars}")
        print(f"Range: {self.input_range}, front factor: {self.input_range_factor_front}")
        print(f'Loading model from {LOAD_CKPT_PATH}')

        if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
            self.net = LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH, map_location=self.device)
        else:
            raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        self.net.eval()

        self.cleared_stop_sign = False
        self.moving_walkers = set()

        self.lat_pid = LateralPIDController(self.config)
        self.lon_pid = LongitudinalLinearRegressionController(self.config)

        self.plant_vars = PlanTVariables()
        self.speed_cats = self.plant_vars.speed_cats
        self.bev_colors = torch.tensor(self.plant_vars.bev_colors)

    def _init(self, hd_map, vehicle):
        super()._init(hd_map, vehicle)

        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        if self.input_bev:
            obs_config = {
                'width_in_pixels': self.config.lidar_resolution_width,
                'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
                'pixels_per_meter': self.config.pixels_per_meter_collection,
                'history_idx': [-1],
                'scale_bbox': True,
                'scale_mask_col': 1.0,
                'map_folder': 'maps_2ppm_cv'
            }

            self.ss_bev_manager = ObsManager(obs_config, self.config)
            self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

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
        }, {
            "type": "sensor.speedometer",
            "reading_frequency": 20,
            "id": "speed"
        },{
            'type': 'sensor.other.gnss',
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'sensor_tick': 0.01,
            'id': 'gps'
            }]
        
        if self.visualize_plant and self.use_rgb:
            result.append({
            'type': 'sensor.camera.rgb',
            'x': self.config.camera_pos[0],
            'y': self.config.camera_pos[1],
            'z': self.config.camera_pos[2],
            'roll': self.config.camera_rot_0[0],
            'pitch': self.config.camera_rot_0[1],
            'yaw': self.config.camera_rot_0[2],
            'width': 1024, # self.config.camera_width,
            'height': 512, # self.config.camera_height,
            'fov': self.config.camera_fov,
            'id': 'rgb'
            })

        return result

    def tick(self, input_data):
        result = {}

        loc = self._vehicle.get_location()
        pos = np.array([loc.x, loc.y, loc.z])
        speed = input_data['speed'][1]['speed']
        compass = t_u.preprocess_compass(input_data['imu'][1][-1])

        if self.visualize_plant and self.use_rgb:
            result["rgb"] = input_data["rgb"]
        result["speed"] = speed
        result["yaw"] = t_u.normalize_angle(compass)

        result['gps'] = pos[:2]

        if self.input_bev:
            bev = self.ss_bev_manager.get_observation(None)['bev_semantic_classes']
            bev = np.rot90(bev)
            bev = self.bev_colors[torch.tensor(bev[64:-64, 64:-64].copy(), dtype=torch.int)].permute(2, 0, 1)
            result["BEV"] = bev

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None, vehicle=None):
        self.step += 1
        if not self.initialized:
            self._init(None, vehicle)

        tick_data = self.tick(input_data)

        # Route wps
        self._waypoint_planner.load()
        _, _, _, next_light_dist, next_traffic_light, next_stop_dist, next_stop_sign, speed_limit = self._waypoint_planner.run_step(tick_data["gps"])
        waypoint_route = self._waypoint_planner.original_route_points[self._waypoint_planner.route_index:][self.config.tf_first_checkpoint_distance:][::self.config.points_per_meter]
        self.waypoint_route = waypoint_route[:20, :2]
        self._waypoint_planner.save()

        tick_data["speed_limit"] = self.speed_cats[round(speed_limit*3.6)]

        tick_data["route"] = np.array([t_u.inverse_conversion_2d(p, tick_data['gps'], tick_data["yaw"]) for i, p in enumerate(self.waypoint_route)])

        # Boxes
        label_raw = self.get_bounding_boxes()

        for x in label_raw:
            if "position" in x:
                pos_x, pos_y, pos_z = x["position"]
                x_div = self.input_range_factor_front**2 if pos_x > 0 else 1
                if pos_x**2/x_div + pos_y**2 > self.input_range**2 or abs(pos_z) > 30:
                    x["class"] = "too far"

        ego_vehicle_location = self._vehicle.get_location()
        ego_transform = self._vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_vehicle_speed = self._vehicle.get_velocity().length()

        # Traffic lights
        if next_traffic_light is not None and next_light_dist < 30:
            for light, _, waypoints in self.list_traffic_lights:
                if light.id != next_traffic_light.id:
                    continue

                global_rot = light.get_transform().rotation
                relative_yaw = t_u.normalize_angle(np.deg2rad(global_rot.yaw) - ego_yaw)
                for wp in waypoints:
                    relative_pos = t_u.get_relative_transform(ego_matrix, np.array(wp.transform.get_matrix()))
                    label_raw.append({
                        'class': 'traffic_light',
                        'extent': [1.5, 1.5, 0.5],
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw,
                        'state': str(light.state)
                        })

        # Stop sign
        if next_stop_sign is not None and not self.cleared_stop_sign and next_stop_dist < 30:
            center_bb_stop_sign = next_stop_sign.get_transform().transform(next_stop_sign.trigger_volume.location)
            stop_wp = self.world_map.get_waypoint(center_bb_stop_sign)
            rotation_stop_sign = next_stop_sign.get_transform().rotation
            relative_yaw = t_u.normalize_angle(np.deg2rad(rotation_stop_sign.yaw) - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, np.array(stop_wp.transform.get_matrix()))

            label_raw.append({
                        'class': 'stop_sign',
                        'extent': [1.5, 1.5, 0.5],
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw
                        })

        # Calculate the accurate distance to the stop sign
        if next_stop_sign is not None:
            distance_to_stop_sign = next_stop_sign.get_transform().transform(next_stop_sign.trigger_volume.location) \
                .distance(ego_vehicle_location)
        else:
            distance_to_stop_sign = 999999999

        # Reset the stop sign flag if we are farther than 10m away
        if distance_to_stop_sign > self.config.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
        else:
            # Set the stop sign flag if we are closer than 3m and speed is low enough
            if ego_vehicle_speed < 0.1 and distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
                self.cleared_stop_sign = True

        self.control = self._get_control(label_raw, tick_data)

        inital_frames_delay = 40
        if self.step < inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        return self.control

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

    def _get_control(self, label_raw, input_data):
        gt_velocity = input_data['speed'] # torch.FloatTensor([input_data['speed']]).unsqueeze(0)
        input_batch = self.get_input_batch(label_raw, input_data)

        for x in input_batch:
            input_batch[x] = input_batch[x].to(self.device) # Does it work inplace?

        input_batch["y_objs"] = None

        (pred_path, pred_wps, pred_speed) = self.net(input_batch)[2]

        if pred_path is not None:
            pred_path = pred_path.detach().squeeze().cpu().numpy()
        if pred_wps is not None:
            pred_wps = pred_wps.detach().squeeze().cpu().numpy()

        if pred_speed is not None:
            pred_speed = pred_speed.detach().squeeze().cpu()
            pred_speed = F.softmax(pred_speed, dim=0)
            pred_speed = pred_speed.numpy()
            pred_speed = np.array([0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20]) * pred_speed
            desired_speed = sum(pred_speed)
        else:
            desired_speed = np.linalg.norm((pred_wps[2] - pred_wps[3])) * 4.0 # Using 3rd and 4th waypoint for speed

            # Creep heuristic
            mean_speed = np.linalg.norm(pred_wps[:-1] - pred_wps[1:], axis=-1).mean() * 4.0
            if gt_velocity < 0.01:
                desired_speed = min(mean_speed, 0.1)

        throttle, brake = self.lon_pid.get_throttle_and_brake(desired_speed < 0.05, desired_speed, gt_velocity)

        #### Steering 
        if pred_path is None:
            interp_wp = self.interpolate_waypoints(pred_wps)
        else:
            interp_wp = self.interpolate_waypoints(pred_path)

        if gt_velocity < 0.05 and brake:
            # Integral accumulation
            steer = self.lat_pid.step(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]), gt_velocity, np.array([0., 0.]), 0., False)
        else:
            steer = self.lat_pid.step(interp_wp, gt_velocity, np.array([0., 0.]), 0., False)

        self.control_history.append((float(steer), float(throttle), float(brake)))
        self.control_history = self.control_history[1:]

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        viz_trigger = self.step % 5 == 0 and self.visualize_plant
        if viz_trigger and self.step > 2:
            for x in input_batch:
                if input_batch[x] is None:
                    continue
                input_batch[x] = input_batch[x].cpu()

            if pred_wps is not None:
                input_batch["waypoints"] = [pred_wps]
            else:
                input_batch["waypoints"] = [[]]

            if pred_path is not None:
                input_batch["pred_path"] = [pred_path]
            else:
                input_batch["pred_path"] = [[]]

            record = {"boxes": input_batch["x_objs"][2:].tolist(), # Fine for agent
                    "ego_pos": input_data["gps"].tolist(),
                    "ego_rot": input_data["yaw"],
                    "waypoints": np.array(input_batch["waypoints"][0]).tolist(),
                    "route_original": np.array(input_batch["route_original"][0]).tolist(),
                    "route": np.array(input_batch["pred_path"][0]).tolist(),
                    "control_history": self.control_history,
                    "frame": GameTime.get_frame(),
                    "ego_speed": round(input_data["speed"], 2)}

            with open(f"{self.img_path}.txt", "a") as f:
                line = json.dumps(record) + "\n"
                line = re.sub(r'(\d+\.\d{3})\d*', r'\1', line)
                f.write(line)

            if self.use_rgb and self.step % 20 == 0:
                img = viz_batch(input_batch, rgb=input_data["rgb"], range_front=self.input_range*self.input_range_factor_front, range_sides=self.input_range)#, control_history=self.control_history) #, input_ego=self.cfg_agent.model.training.input_ego)
                cv2.imwrite(f"{self.img_path}/{GameTime.get_frame()}.jpg", img)

        return control
    
    
    def get_input_batch(self, label_raw, input_data):
        sample = {'input': [], 'output': [], 'route': [], 'waypoints': [], 'target_point': []}

        car_types = self.plant_vars.car_types
        type_nums = self.plant_vars.class_nums

        # Statics don't appear in longest6
        if self.route_index and "longest6" in self.route_index:
            type_nums.pop("static", None)

        if not self.input_static_cars:
            type_nums.pop("static_car", None)
        
        for x in label_raw:
            # We skip walkers that haven't moved yet, but keep walkers that have already moved
            # (Walkers may run into a vehicle, stop, be removed from input because they dont move, plant causes crash)
            if x["class"] == "walker":
                if x["speed"] < 0.1 and x["id"] not in self.moving_walkers:
                    x["class"] = "irrelevant_walker"
                else:
                    self.moving_walkers.add(x["id"])
            
            elif x["class"] == "car" and x["type_id"] in ["vehicle.dodge.charger_police",
                                                        "vehicle.dodge.charger_police_2020",
                                                        "vehicle.carlamotors.firetruck",
                                                        "vehicle.ford.ambulance"]:
                x["class"] = "emergency"

            elif x["class"] == "static":
                if "type_id" in x.keys() and x["type_id"] not in ["static.prop.constructioncone", 
                                                                    "static.prop.trafficwarning"]:
                    x["class"] = "irrelevant_static"
                else:
                    # # update static extent
                    if x["type_id"] in STATIC_EXTENTS:
                        x["extent"] = STATIC_EXTENTS[x["type_id"]]
                    else:
                        print(x["type_id"], "was not found in static extents")

            elif x["class"] == "static_car":
                if x["mesh_path"] in CAR_EXTENTS:
                    x["extent"] = CAR_EXTENTS[x["mesh_path"]]
                    if "scale" in x.keys() and x["scale"] is not None:
                        scale = float(x["scale"])
                        x["extent"] = [a*scale for a in x["extent"]]
                else:
                    print("missing static car:", x["mesh_path"])

        data_car = [
            [
                type_nums[x["class"].lower()],  # type indicator
                x['position'][0],
                x['position'][1],
                rad2deg(x['yaw']), # in degrees
                x['speed'] * 3.6, # in km/h
                x['extent'][1]*2 + (0 if "scenario" not in x.keys() or "Door" not in x["scenario"] else 1),
                x['extent'][0]*2
            ]
            for j, x in enumerate(label_raw)
            if x["class"].lower() in car_types
        ]

        data_car += [[
                type_nums[x["class"].lower()], # type indicator
                x['position'][0],
                x['position'][1],
                rad2deg(x['yaw']), # in degrees
                0.0,
                x['extent'][1]*2,
                x['extent'][0]*2
            ]
            for j, x in enumerate(label_raw)
            if x["class"].lower() not in car_types and x["class"].lower() in type_nums.keys() and (not x["class"].lower()=="traffic_light" or x["state"] in ["Red", "Yellow"])
        ] 

        features = data_car

        sample['input'] = features
        sample["route_original"] = input_data["route"]
        sample["speed_limit"] = input_data["speed_limit"]
        sample["ego_speed"] = input_data["speed"]

        if self.input_bev:
            sample["BEV"] = input_data["BEV"]

        # dummy data
        sample['output'] = []
        sample["waypoints"] = 0
        sample["route"] = []
        sample["target_speed"] = 0

        batch = [sample]

        input_batch = generate_batch(batch)

        return input_batch

    def destroy(self, results = None):
        super().destroy()
        del self.net