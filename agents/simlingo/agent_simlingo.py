"""
partially taken from https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/sensor_agent.py
(MIT licence)
"""


import importlib.util
import json
import math
import os
import pathlib
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import carla
import cv2
import hydra
import numpy as np
import torch
import ujson
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from hydra.utils import get_original_cwd, to_absolute_path
from leaderboardcodes import autonomous_agent2
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve
from transformers import AutoConfig, AutoProcessor

import scenario_logger
import transfuser_utils as t_u
from scenario_logger import ScenarioLogger
from custom_types import DrivingInput, LanguageLabel
from internvl2_utils import build_transform, dynamic_preprocess
from config_simlingo import GlobalConfig
from nav_planner import LateralPIDController, RoutePlanner
from simlingo_utils import (
    get_camera_extrinsics,
    get_camera_intrinsics,
    get_rotation_matrix,
    project_points,
)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


# Leaderboard function that selects the class used as agent.
def get_entry_point():
    return 'LingoAgent'


DEBUG = False # saves images during evaluation
HD_VIZ = False
USE_UKF = True

class LingoAgent(autonomous_agent2.AutonomousAgent):
    """
        Main class that runs the agents with the run_step function
        """

    def setup(self, path_to_conf_file, route_index=None):
        """Sets up the agent. route_index is for logging purposes"""

        torch.cuda.empty_cache()
        self.track = autonomous_agent2.Track.SENSORS
        if '+' in path_to_conf_file:
            print(f"path to conf file: {path_to_conf_file}")
            self.config_path = path_to_conf_file.split('+')[0]
            print(f"Config path: {self.config_path}")
            self.save_path_root = path_to_conf_file.split('+')[1]
            print(f"Save path root: {self.save_path_root}")
        else:
            self.config_path = path_to_conf_file
            print(f"Config path: {self.config_path}")
            self.save_path_root = route_index
            print(f"Save path root: {self.save_path_root}")
        self.step = -1
        self.initialized = False
        self.device = torch.device('cuda')
        self.DrivingInput = {}
        self.config = GlobalConfig()

        if self.config.eval_route_as == -1:
            self.config.eval_route_as = self.model.route_as

        self.last_command = -1
        self.last_command_tmp = -1
        self.user_command = None
        self.user_flag = None
        self.running = True
        self.custom_prompt = None
        
        self.LMDRIVE_AUGM = False
        if self.LMDRIVE_AUGM:
                command_templates_file = f"data/augmented_templates/lmdrive.json"
                with open(command_templates_file, 'r') as f:
                        self.command_templates = ujson.load(f)
        
        # used for interactive eval of instruction following
        # thread = threading.Thread(target=self.input_thread)
        # thread.daemon = True  # This makes the thread exit when the main program exits
        # thread.start()

        self.route_path = os.environ.get('ROUTES', '')
        route_type = self.route_path.split('data/benchmarks/')[-1].split('/')[0]
        route_number = str(pathlib.Path(self.route_path).stem)


        # PID controller for turning - used in earlier versions of the agent
        # self.turn_controller = t_u.PIDController(k_p=self.config.turn_kp,
        #                                          k_i=self.config.turn_ki,
        #                                          k_d=self.config.turn_kd,
        #                                          n=self.config.turn_n)
        self.speed_controller = t_u.PIDController(k_p=self.config.speed_kp,
                                                                                            k_i=self.config.speed_ki,
                                                                                            k_d=self.config.speed_kd,
                                                                                            n=self.config.speed_n)

        self.turn_controller = LateralPIDController(inference_mode=False)

        image_fps = 5
        image_history_length = 1

        self.image_buffer = deque(maxlen=image_fps * image_history_length)

        # config
        self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
        self.data_save_freq = 5
        self.lidar_seq_len = 1
        self.logging_freq = 10  # Log every 10 th frame
        self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
        self.dense_route_planner_min_distance = 1.0
        self.dense_route_planner_max_distance = 50.0
        self.log_route_planner_min_distance = 4.0
        self.route_planner_max_distance = 50.0
        self.route_planner_min_distance = 7.5

        #load config from .hydra folder
        self.config_load_path = Path(self.config_path).parent.parent.parent / '.hydra'/ 'config.yaml'
        with open(self.config_load_path, 'r') as file:
            cfg = OmegaConf.load(file)
        self.cfg = cfg
        self.cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
    
        processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
        if 'tokenizer' in processor.__dict__:
                self.tokenizer = processor.tokenizer
        else:
                self.tokenizer = processor
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"
        # llm_tokenizer = AutoTokenizer.from_pretrained(cfg.model.language_model.variant)
        cache_dir = f"pretrained/{(cfg.model.vision_model.variant.split('/')[1])}"
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=cache_dir,
                _recursive_=False
            ).to(self.device)
        torch.set_default_dtype(default_dtype)
        self.model.load_state_dict(torch.load(self.config_path))
        self.iter = self.config_path.split("epoch=")[-1].split("/")[0]
        self.session = self.config_path.split("/")[-4]
        
        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0

        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Filtering
        if USE_UKF:
            self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
            self.ukf = UKF(dim_x=4,
                                        dim_z=4,
                                        fx=bicycle_model_forward,
                                        hx=measurement_function_hx,
                                        dt=self.carla_frame_rate,
                                        points=self.points,
                                        x_mean_fn=state_mean,
                                        z_mean_fn=measurement_mean,
                                        residual_x=residual_state_x,
                                        residual_z=residual_measurement_h)

            # State noise, same as measurement because we
            # initialize with the first measurement later
            self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
            # Measurement noise
            self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
            self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
            # Used to set the filter state equal the first measurement
            self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
        self.state_log = deque(maxlen=max((self.lidar_seq_len * self.data_save_freq), 2))

        # Path to where visualizations and other debug output gets stored
        if self.save_path_root is not None:
            self.save_path = os.environ.get('SAVE_PATH', '') + self.save_path_root
        else:
            self.save_path = None
        # self.checkpoint_path = os.environ.get('CHECKPOINT_ENDPOINT').

        # Logger that generates logs used for infraction replay in the results_parser.
        if self.save_path is not None and route_index is not None:
            self.save_path = pathlib.Path(self.save_path) / route_index
            pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

            self.lon_logger = ScenarioLogger(
                    save_path=self.save_path,
                    route_index=route_index,
                    logging_freq=self.logging_freq,
                    log_only=True,
                    route_only=False,  # with vehicles
                    roi=self.logger_region_of_interest,
            )
        
        if self.save_path is not None:
            self.debug_save_path = self.save_path + '/debug_viz' + f'/{self.session}/iter_{self.iter}/{route_type}/{route_number}_{time.strftime("%Y_%m_%d_%H_%M_%S")}'
            Path(self.debug_save_path).mkdir(parents=True, exist_ok=True)
            self.save_path_metric = self.debug_save_path + '/metric'
            Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        if DEBUG:
            self.save_path_img = self.debug_save_path + '/images'
            Path(self.save_path_img).mkdir(parents=True, exist_ok=True)
            
    def input_thread(self):
        while self.running:
            user_input = input("Enter a command for the vehicle. 1: turn left, 2: turn right, 3: lane change left, 4: lane change right, 5: stop, 6: accelerate: ")
            if user_input.isdigit():
                    self.user_flag = int(user_input)
                # if int(user_input) == 1:
                #   self.user_command = 'turn left at the next intersection'
                # elif int(user_input) == 2:
                #   self.user_command = 'turn right at the next intersection'
                # elif int(user_input) == 3:
                #   self.user_command = 'change one lane to the left'
                # elif int(user_input) == 4:
                #   self.user_command = 'change one lane to the right'
                # elif int(user_input) == 5:
                #   self.user_command = 'stop'
                # elif int(user_input) == 6:
                #   self.user_command = 'accelerate'
                    
            else:
                self.user_command = str(user_input)
                
            if user_input.strip().lower() == "exit":
                self.running = False
            
            print(f"User command: {self.user_command}")
            print(f"User flag: {self.user_flag}")

    def _init(self):
        # The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
        # GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0
        # But town 13 has a different value in CARLA 0.9.15. The following code, adapted from Bench2DriveZoo estimates the
        # lat, lon reference values by abusing the fact that the leaderboard exposes the route plan also in CARLA
        # coordinates. The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale
        # of the GPS. It seems to work reasonably well, so we use this workaround for now.
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation
            def equations(variables):
                x, y = variables
                eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
                             - math.cos(x * math.pi / 180.0) * y)
                eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
                             * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
                             * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
                return [eq1, eq2]
            initial_guess = [0.0, 0.0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0.0, 0.0
        self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
                                                                             self.lat_ref, self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors = []
        for num_cam in self.config.num_cameras:
            # get from config by name as string
            sensors += [
                    {
                            'type': 'sensor.camera.rgb',
                            'x': self.config.__dict__[f'camera_pos_{num_cam}'][0],
                            'y': self.config.__dict__[f'camera_pos_{num_cam}'][1],
                            'z': self.config.__dict__[f'camera_pos_{num_cam}'][2],
                            'roll': self.config.__dict__[f'camera_rot_{num_cam}'][0],
                            'pitch': self.config.__dict__[f'camera_rot_{num_cam}'][1],
                            'yaw': self.config.__dict__[f'camera_rot_{num_cam}'][2],
                            'width': self.config.__dict__[f'camera_width_{num_cam}'],
                            'height': self.config.__dict__[f'camera_height_{num_cam}'],
                            'fov': self.config.__dict__[f'camera_fov_{num_cam}'],
                            'id': f'rgb_{num_cam}'
                    }
            ]

        if HD_VIZ:
            sensors += [{
                                                'type': 'sensor.camera.rgb',
                                                'x': -5.5, 'y': 0.0, 'z':3.5,
                                                'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                                                # 'width': 960, 'height': 540, 'fov': 110,
                                                # 'width': 1280, 'height': 720, 'fov': 120,
                                                'width': 1920, 'height': 1080, 'fov': 110,
                                                'id': 'rgb_viz'
            }]

        sensors += [{
                'type': 'sensor.other.imu',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': self.config.carla_frame_rate,
                'id': 'imu'
        }, {
                'type': 'sensor.other.gnss',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
        }, {
                'type': 'sensor.speedometer',
                'reading_frequency': self.config.carla_fps,
                'id': 'speed'
        }, 
        ]

        return sensors

    @torch.inference_mode()  # Turns off gradient computation
    def tick(self, input_data):
        """Pre-processes sensor data and runs the Unscented Kalman Filter"""
        rgb = []

        if HD_VIZ:
            self.hd_cam_for_viz = input_data['rgb_viz'][1][:, :, :3]

        for camera_pos in self.config.num_cameras:
            rgb_cam = 'rgb_' + str(camera_pos)
            camera = input_data[rgb_cam][1][:, :, :3]
            if camera_pos == 0:
                self.camera_for_viz = camera.copy()

            # Also add jpg artifacts at test time, because the training data was saved as jpg.
            _, compressed_image_i = cv2.imencode('.jpg', camera)
            camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

            rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb_pos = rgb_pos[:int(rgb_pos.shape[0] - (rgb_pos.shape[0] * 4.8) // 16), :, :] # do this from config to ensure it is the same as in training

            # Switch to pytorch channel first order
            rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
            rgb.append(rgb_pos)

        rgb = np.array(rgb)
        self.image_buffer.append(rgb)

        rgbs = rgb
        image_sizes = None
        
        if 'internvl2' in self.cfg.model.vision_model.variant.lower():
            T, C, H, W = rgbs.shape
            transform = build_transform(input_size=448)
            images_processed_tmp = []
            images_sizes_tmp = []
            
            image = Image.fromarray(rgbs.squeeze(0).transpose(1, 2, 0))
            images = dynamic_preprocess(image, image_size=448, use_thumbnail=self.cfg.model.vision_model.use_global_img, max_num=2)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            images_processed_tmp.append(pixel_values)
            images_sizes_tmp.append([image.size[1], image.size[0]])
            
            images_processed = {
                    'pixel_values': torch.stack(images_processed_tmp), 
                    'image_sizes': torch.tensor(images_sizes_tmp)
                    }  
            processed_image = images_processed['pixel_values']
            num_patches = processed_image.shape[1]
            new_height = processed_image.shape[3]
            new_width = processed_image.shape[4]
            processed_image = processed_image.view(1, self.T, num_patches, C, new_height, new_width)
            
        else:
            raise NotImplementedError(f"Encoder {self.cfg.data_module.encoder} not implemented yet")
        
        gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
        
        compass = t_u.preprocess_compass(input_data['imu'][1][-1])

        result = {
                'rgb': rgb,
                'compass': compass,
        }
        speed = input_data['speed'][1]['speed']

        if USE_UKF:
            if not self.filter_initialized:
                self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
                self.filter_initialized = True

            self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
            self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
            filtered_state = self.ukf.x

            self.state_log.append(filtered_state)
            result['gps'] = filtered_state[0:2]
        else:
            result['gps'] = np.array([gps_pos[0], gps_pos[1]])
            
        speed = round(input_data['speed'][1]['speed'], 1)

        waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))

        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[1]
        else:
            target_point, far_command = waypoint_route[0]
            next_target_point, next_far_command = waypoint_route[0]
            
            
        if self.last_command_tmp != far_command:
            self.last_command = self.last_command_tmp
        
        self.last_command_tmp = far_command
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)

        one_hot_command = t_u.command_to_one_hot(self.commands[-2])
        result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_target_point_torch = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])

        result['target_point'] = ego_target_point_torch

        self.target_points = None
        placeholder_batch_list = []

        if self.config.eval_route_as == 'target_point' or self.config.eval_route_as == 'target_point_command':
            target_points = [ego_target_point, ego_next_target_point]
            self.target_points = target_points.copy()
            target_points_np = np.array(target_points)
            target_points = torch.from_numpy(target_points_np).to(self.device, dtype=torch.float32).unsqueeze(0)
            result['route'] = target_points
            
            placeholder_values = {'<TARGET_POINT>': target_points_np}
            tmp = {}
            for key, value in placeholder_values.items():
                    token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                    tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)
            
            prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
            
        elif self.config.eval_route_as == 'command':
            # get distance from target_point
            dist_to_command = np.linalg.norm(ego_target_point)
            dist_to_command = int(dist_to_command)
            map_command = {
                    1: 'go left at the next intersection',
                    2: 'go right at the next intersection',
                    3: 'go straight at the next intersection',
                    4: 'follow the road',
                    5: 'do a lane change to the left',
                    6: 'do a lane change to the right',        
            }
            command_template_mappings = {
                    1: [0, 2, 4, 7],
                    2: [1, 3, 5, 8],
                    3: [6, 9],
                    4: [38, 40, 42, 43, 44, 45],
                    5: [34, 36],
                    6: [35, 37],
            }
            if self.LMDRIVE_AUGM:
                lmdrive_index = random.choice(command_template_mappings[far_command])
                lmdrive_command = random.choice(self.command_templates[str(lmdrive_index)])
                lmdrive_command = lmdrive_command.replace('[x]', str(dist_to_command))
                prompt_tp = f'Command: {lmdrive_command}'
                
            else:
                command = map_command[far_command]
                next_command = map_command[next_far_command]
                if self.last_command in [1, 2, 3] and far_command == 4:
                    next_command = command
                    command = map_command[self.last_command]
                    
                if command != next_command:
                        next_command = f' then {next_command}'
                else:
                        next_command = ''
                        
                if far_command == 4:
                        prompt_tp = f'Command: {command}{next_command}.'
                else:
                        prompt_tp = f'Command: {command} in {dist_to_command} meter{next_command}.'
                
        else:
            result['route'] = route_img

        if self.config.use_cot:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
        else:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} Predict the waypoints."
        
        if self.custom_prompt is not None:
            if self.user_flag == 2 or self.user_flag == 3:
                prompt = f"Current speed: {speed} m/s. {self.custom_prompt}"
            else:
                prompt = f"Current speed: {speed} m/s. {prompt_tp} {self.custom_prompt}"


        if self.user_flag == 1 or self.user_flag == 2:
            prompt = f"<INSTRUCTION_FOLLOWING> {prompt}"
        elif self.user_flag == 0:
            prompt = f"<SAFETY> {prompt}"


        result['speed'] = torch.FloatTensor([speed]).unsqueeze(0).to(self.device, dtype=torch.float32)

        B, T, num_patches, C, H, W = processed_image.shape
        assert B == 1
        assert T == self.T
        assert C == 3

        speed = round(speed, 1)
        
        self.prompt_tp = prompt_tp
        self.prompt = prompt
        
        conversation_all = [
                {
                "role": "user",
                "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                        ],
                },
                {
                "role": "assistant",
                "content": [
                        {"type": "text", "text": "Waypoints:"},
                        ],
                },
        ]
        conv_batch_list = [conversation_all]
        questions = []
        for conv in conv_batch_list:
                for i in range(len(conv)):
                        questions.append(conv[i]['content'][0]['text'])
                        conv[i]['content'] = conv[i]['content'][0]['text']
                        
        cache_dir = f"pretrained/{(self.cfg.model.vision_model.variant.split('/')[1])}"
        # get absolute path from workspace dir not wokring dir
        cache_dir = to_absolute_path(cache_dir)
        model_path = f"{cache_dir}/conversation.py"
        if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=self.cfg.model.vision_model.variant, local_dir=cache_dir)
                
        #import from file from model_path
        spec = importlib.util.spec_from_file_location('get_conv_template', model_path)
        conv_module = importlib.util.module_from_spec(spec)
        sys.modules['get_conv_template'] = conv_module
        spec.loader.exec_module(conv_module)
        
        if not hasattr(self, 'tmp_config'):
                self.tmp_config = AutoConfig.from_pretrained(self.cfg.model.vision_model.variant, trust_remote_code=True)
                image_size = self.tmp_config.force_image_size or self.tmp_config.vision_config.image_size
                patch_size = self.tmp_config.vision_config.patch_size
                
                self.num_image_token = int((image_size // patch_size) ** 2 * (self.tmp_config.downsample_ratio ** 2))
                
        prompt_batch_list = []
        for idx, conv in enumerate(conv_batch_list):
                question = questions[idx]
                if '<image>' not in question:
                        question = '<image>\n' + question
                template = conv_module.get_conv_template('internlm2-chat')
                template_inference = None
                
                template_inference = conv_module.get_conv_template('internlm2-chat')
                for conv_part_idx, conv_part in enumerate(conv):
                        if conv_part['role'] == 'assistant':
                                # template.append_message(template.roles[1], conv_part['content'])
                                template.append_message(template.roles[1], None)
                        elif conv_part['role'] == 'user':
                                if conv_part_idx == 0 and '<image>' not in conv_part['content']:
                                        # add image token
                                        conv_part['content'] = '<image>\n' + conv_part['content']
                                template.append_message(template.roles[0], conv_part['content'])
                        else:
                                raise ValueError(f"Role {conv_part['role']} not supported")
                            
                query = template.get_prompt()
                # remove system prompt
                system_prompt = template.system_template.replace('{system_message}', template.system_message) + template.sep
                query = query.replace(system_prompt, '')
                
                IMG_START_TOKEN='<img>'
                IMG_END_TOKEN='</img>'
                IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
                num_patches_all = 2 # sum(grid_nums)

                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches_all + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
                prompt_batch_list.append(query)
                
        prompt_tokenized = self.tokenizer(prompt_batch_list, padding=True, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        prompt_tokenized_ids = prompt_tokenized["input_ids"]
        prompt_tokenized_char_offsets = prompt_tokenized["offset_mapping"].view(1, -1, 2)
        prompt_tokenized_valid = prompt_tokenized["input_ids"] != self.tokenizer.pad_token_id
        prompt_tokenized_mask = prompt_tokenized_valid
        
        ll = LanguageLabel(
                phrase_ids=prompt_tokenized_ids.to(self.device),
                phrase_valid=prompt_tokenized_valid.to(self.device),
                phrase_mask=prompt_tokenized_mask.to(self.device),
                placeholder_values=placeholder_batch_list,
                language_string=prompt_batch_list,
                loss_masking=None,
        )

        self.DrivingInput["camera_images"] = processed_image.to(self.device).bfloat16()
        self.DrivingInput["image_sizes"] = image_sizes
        self.DrivingInput["camera_intrinsics"] = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), 1, dim=0).view(1, 3, 3).float().to(self.device),
        self.DrivingInput["camera_extrinsics"] = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), 1, dim=0).view(1, 4, 4).float().to(self.device),
        self.DrivingInput["vehicle_speed"] = result['speed']
        self.DrivingInput["target_point"] = result['target_point'].to(self.device)
        self.DrivingInput["prompt"] = ll
        self.DrivingInput["prompt_inference"] = ll

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            return control

        # Need to run this every step for GPS filtering
        tick_data = self.tick(input_data)

        # initialize DrivingInput with dict self.DrivingInput
        model_input = DrivingInput(**self.DrivingInput)
        pred_speed_wps, pred_route, language = self.model(model_input)
        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None

        # prepare velocity input
        gt_velocity = tick_data['speed']

        if DEBUG and self.step%5 == 0:
            tvec = None
            rvec = None

            if HD_VIZ:
                self.camera_for_viz = self.hd_cam_for_viz
                tvec = np.array([[0.0, 3.5, 5.5]], np.float32)

                cam_rots = [0.0, -15.0, 0.0]
                rot_matrix = get_rotation_matrix(-cam_rots[0], -cam_rots[1], cam_rots[2])
                rvec = cv2.Rodrigues(rot_matrix[:3, :3])[0].flatten()

            W=self.camera_for_viz.shape[1]
            H=self.camera_for_viz.shape[0]
            camera_intrinsics = np.asarray(get_camera_intrinsics(W,H,110))

            # bgr to rgb
            self.camera_for_viz = cv2.cvtColor(self.camera_for_viz, cv2.COLOR_BGR2RGB)

            # draw the predicted waypoints
            image = Image.fromarray(self.camera_for_viz)
            draw = ImageDraw.Draw(image)

            if self.target_points is not None:
                target_point_img_coords = project_points(self.target_points, camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in target_point_img_coords:
                    # in blue
                    draw.ellipse((points_2d[0]-4, points_2d[1]-4, points_2d[0]+4, points_2d[1]+4), fill=(0, 0, 255, 255))

            if pred_route is not None:
                pred_route_img_coords = project_points(pred_route[0].detach().cpu().numpy(), camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in pred_route_img_coords:
                        draw.ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(255, 0, 0, 255))
            
            if pred_speed_wps is not None:
                pred_speed_wps_img_coords = project_points(pred_speed_wps[0].detach().cpu().numpy(), camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in pred_speed_wps_img_coords:
                        draw.ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(0, 255, 0, 255))

            if language is not None:
                # write the language to the bottom of the image
                black_box = Image.new('RGBA', (W, 400), (0, 0, 0, 255))
                # concatenate the images
                image_all = Image.new('RGBA', (W, H+400))
                image_all.paste(image, (0, 0))
                image_all.paste(black_box, (0, H))
                image = image_all
                draw = ImageDraw.Draw(image)

                if HD_VIZ:
                    font_size = 50
                    line_width = 60
                    y_dist = 60
                    y_start = H + 20
                else:
                    font_size = 20
                    line_width = 100
                    y_dist = 30
                    y_start = H + 20
                font = ImageFont.truetype("arial.ttf", font_size)
                import textwrap
                lines = textwrap.wrap(f"Prompt: {self.prompt}", width=line_width)
                for idx, line in enumerate(lines):
                        draw.text((10, y_start + y_dist*(idx)), line, font=font, fill=(255, 255, 255, 255))
                
                y_start = H + 20 + y_dist*(idx+1)

                lines = textwrap.wrap(f"Answer: {language[0]}", width=line_width)
                for idx, line in enumerate(lines):
                        draw.text((10, y_start + y_dist*(idx)), line, font=font, fill=(255, 255, 255, 255))

            # save
            image.save(f"{self.save_path_img}/{self.step}.png")
            
        steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)

        # # 0.1 is just an arbitrary low number to threshold when the car is stopped
        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1
            print(f"force_move: {self.force_move}")

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        # CARLA will not let the car drive in the initial frames.
        # We set the action to brake so that the filter does not get confused.
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control
        
        #metric_info = self.get_metric_info()
        #self.metric_info[self.step] = metric_info
        #if self.save_path_metric is not None and self.step % 1 == 0:
                # metric info
                #outfile = open(f"{self.save_path_metric}/metric_info.json", 'w')
                #json.dump(self.metric_info, outfile, indent=4)
                #outfile.close()

        return control

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        """
        Predicts vehicle control with a PID controller.
        Used for waypoint predictions
        """
        assert route_waypoints.size(0) == 1
        route_waypoints = route_waypoints[0].data.cpu().numpy()
        speed = velocity[0].data.cpu().numpy()
        speed_waypoints = speed_waypoints[0].data.cpu().numpy()

        # m / s required to drive
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]) * 2.0

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        route_interp = self.interpolate_waypoints(route_waypoints.squeeze())

        steer = self.turn_controller.step(route_interp, speed)

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer, throttle, brake
    
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
    
    def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
        """
        Gets called after a route finished.
        The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
        Also writes logging files to disk.
        """

        del self.model
        del self.config
        if self.cfg.data_module.encoder == 'llavanext':
            del self.processor


# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
    # Kinematic bicycle model.
    # Numbers are the tuned parameters from World on Rails
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x


def measurement_function_hx(vehicle_state):
    '''
        For now we use the same internal state as the measurement state
        :param vehicle_state: VehicleState vehicle state variable containing
                                                    an internal state of the vehicle from the filter
        :return: np array: describes the vehicle state as numpy array.
                                             0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
        '''
    return vehicle_state


def state_mean(state, wm):
    '''
        We use the arctan of the average of sin and cos of the angle to calculate
        the average of orientations.
        :param state: array of states to be averaged. First index is the timestep.
        :param wm:
        :return:
        '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def measurement_mean(state, wm):
    '''
    We use the arctan of the average of sin and cos of the angle to
    calculate the average of orientations.
    :param state: array of states to be averaged. First index is the
    timestep.
    '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y
