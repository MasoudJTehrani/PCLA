"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""
import datetime
import glob
import gzip
import os
import pickle as pkl
import random
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import ujson
from hydra.utils import get_original_cwd
from imgaug import augmenters as ia
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from tqdm import tqdm

import simlingo_training.utils.transfuser_utils as t_u
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.utils.projection import get_camera_intrinsics, project_points

VIZ_DATA = False

class BaseDataset(Dataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Base class for the dataset.
    """

    def __init__(self,
            dreamer = False,
            evaluation = False,
            **cfg,
        ):
        for key, value in cfg.items():
            setattr(self, key, value)

        self.tfs = image_augmenter(prob=self.img_augmentation_prob)

        filter_infractions_per_route = True

        self.rgb_folder = 'rgb'
        self.dreamer_folder = 'dreamer'
        
        self.images = []
        self.boxes = []
        self.measurements = []
        self.sample_start = []
        self.augment_exists = []
        self.alternative_trajectories = []

        self.temporal_measurements = []

        total_routes = 0
        perfect_routes = 0
        crashed_routes = 0

        fail_reasons = {}

        repo_path = get_original_cwd()
        
        # load templates
        template_file = f"{repo_path}/data/augmented_templates/commentary_augmented.json"
        with open(template_file, 'r') as f:
            self.templates_commentary = ujson.load(f)
    
        # load templates
        if dreamer:
            template_file = f"{repo_path}/data/augmented_templates/dreamer.json"
            with open(template_file, 'r') as f:
                self.templates_neg = ujson.load(f)
        
        if self.use_lmdrive_commands:
            command_templates_file = f"{repo_path}/data/augmented_templates/lmdrive.json"
            with open(command_templates_file, 'r') as f:
                self.command_templates = ujson.load(f)


        # during eval we only want to load predefines paths
        if evaluation:
            if self.use_qa:
                chosen_eval_samples_path = f'{repo_path}/data/evalset_vqa.json'
            elif self.use_commentary:
                chosen_eval_samples_path = f'{repo_path}/data/evalset_commentary.json'
            
            with open(chosen_eval_samples_path, 'r') as f:
                self.chosen_eval_samples = ujson.load(f)
                self.all_eval_samples = []
                self.all_eval_samples_dict = {}
                for key, value in self.chosen_eval_samples.items():
                    if self.use_qa:
                        if 'important objects' in key:
                            continue

                        for answer in value.keys():
                            for sample in value[answer]:
                                sample = repo_path + '/' + sample.replace('vqa', 'measurements').replace('drivelm', 'data')
                                self.all_eval_samples.append(sample)
                                
                                if sample not in self.all_eval_samples_dict:
                                    self.all_eval_samples_dict[sample] = [(key, answer)]
                                else:
                                    self.all_eval_samples_dict[sample].append((key, answer))
                    else:
                        for sample in value:
                            sample = repo_path + '/' + sample.replace('commentary/simlingo', 'data/simlingo').replace('commentary', 'measurements')
                            self.all_eval_samples.append(sample)


        augment_exist = False

        if not dreamer:
            if self.use_qa:
                as_augment_file = f'{repo_path}/data/augmented_templates/drivelm_train_augmented_v2/all_as_augmented.json'
                with open(as_augment_file, 'r') as f:
                    self.a_augment = ujson.load(f)
                qs_augment_file = f'{repo_path}/data/augmented_templates/drivelm_train_augmented_v2/all_qs_augmented.json'
                with open(qs_augment_file, 'r') as f:
                    self.q_augment = ujson.load(f)

            prompt_probabilities = {
                'driving': 1.0
            }
            if self.use_qa:
                prompt_probabilities['qa'] = 1.0
            if self.use_commentary:
                prompt_probabilities['commentary'] = 1.0
            
            # divide by the sum to get the probabilities
            prompt_probabilities = {k: v / sum(prompt_probabilities.values()) for k, v in prompt_probabilities.items()}
            self.prompt_probabilities = prompt_probabilities
            self.num_sampled_per_type = {k: 0 for k in prompt_probabilities.keys()}



        if not self.bucket_name == "all":
            with open(f"{repo_path}/" + self.bucket_path + '/buckets_paths.pkl', 'rb') as f:
                bucket_dict = pkl.load(f)

            bucket_run_ids = None

            # TODO: this is stupid that its manual, should change bucket names to match the saved dict with pathes
            if self.bucket_name == "all":
                pass
            elif self.bucket_name == 'acceleration_negative_5':
                bucket_run_ids = bucket_dict['acceleration_-5']# + bucket_dict['acceleration_-20'] + bucket_dict['acceleration_-40']
            elif self.bucket_name == "acceleration_negative_1":
                bucket_run_ids = bucket_dict['acceleration_-1']
            elif self.bucket_name == "acceleration_positive_1":
                bucket_run_ids = bucket_dict['acceleration_5']
            elif self.bucket_name == "acceleration_positive_5":
                bucket_run_ids = bucket_dict['acceleration_20']# + bucket_dict['acceleration_40'] + bucket_dict['acceleration_1000000']
            elif self.bucket_name == "lateral_control_1":
                bucket_run_ids = bucket_dict['lateral_control_1']
            elif self.bucket_name == "lateral_control_1_2":
                bucket_run_ids = bucket_dict['lateral_control_1'] + bucket_dict['lateral_control_2']
            elif self.bucket_name == "lateral_control_high":
                bucket_run_ids = bucket_dict['lateral_control_2'] + bucket_dict['lateral_control_5'] + bucket_dict['lateral_control_1000000']
            elif self.bucket_name == "lateral_control_higher_5":
                bucket_run_ids = bucket_dict['lateral_control_5'] + bucket_dict['lateral_control_1000000']
            elif self.bucket_name == "recovery":
                bucket_run_ids = bucket_dict['recovery_data_small'] + bucket_dict['recovery_data_large']
            else:
                if self.bucket_name not in bucket_dict:
                    raise ValueError(f"Bucket name {self.bucket_name} not found.")
                bucket_run_ids = bucket_dict[self.bucket_name]

            run_id_dict = {}
            if bucket_run_ids is not None:
                for run_id in bucket_run_ids:
                    run_id = run_id.replace('database/simlingo_v2_2025_01_10', self.bucket_path)
                    run_id_path = Path(run_id)
                    run_id_parent = run_id_path.parent
                    run_id_name = run_id_path.name
                    run_id_absolut = str(run_id_parent)
                    run_id_absolut = f"{repo_path}/{str(run_id_parent)}"
                    if run_id_absolut not in run_id_dict:
                        run_id_dict[run_id_absolut] = [run_id_name]
                    else:
                        run_id_dict[run_id_absolut].append(run_id_name)


        route_dirs = glob.glob(f"{repo_path}/" + self.data_path + '/data/simlingo/*/*/*/Town*')
        print(f'Found {len(route_dirs)} routes in {repo_path + self.data_path}')
        
        if not self.use_old_towns:
            route_dirs = [route_dir for route_dir in route_dirs if 'lb1_split' not in route_dir]
            print(f'Found {len(route_dirs)} routes in {repo_path + self.data_path} after filtering out old towns')
        elif self.use_only_old_towns or self.bucket_name == "old_towns":
            route_dirs = [route_dir for route_dir in route_dirs if 'lb1_split' in route_dir]
            print(f'Found {len(route_dirs)} routes in {repo_path + self.data_path} after filtering out non old towns')
        

        random.shuffle(route_dirs)
        split_percentage = 0.99
        if dreamer or not self.use_town13:
            # split the data into official training(Town12 and old Towns) and validation set (Town13)
            if self.split == "train":
                print("Using Town12 for training")
                route_dirs = [route_dir for route_dir in route_dirs if 'routes_training' in route_dir]
            elif self.split == "val":
                print("Using Town13 for validation")
                route_dirs = [route_dir for route_dir in route_dirs if 'routes_validation' in route_dir]
                route_dirs = route_dirs[:int(0.02 * len(route_dirs))]
        else:
            # use all towns
            if self.split == "train":
                route_dirs = route_dirs[:int(split_percentage * len(route_dirs))]
            elif self.split == "val":
                route_dirs = route_dirs[int(split_percentage * len(route_dirs)):]
        
        total_routes += len(route_dirs)
        
        # route_dirs = route_dirs[:100]
        print(f'Use {len(route_dirs)} routes.')
        
        for sub_root in tqdm(route_dirs, file=sys.stdout):

            route_dir = sub_root # + '/' + route
            if dreamer:
                dreamer_dir = route_dir.replace('data/', f'{self.dreamer_folder}/')
                if not os.path.exists(dreamer_dir):
                    continue

            if filter_infractions_per_route:
                if not os.path.isfile(route_dir + '/results.json.gz'):
                    total_routes += 1
                    crashed_routes += 1
                    if "no_results.json" not in fail_reasons:
                        fail_reasons["no_results.json"] = 1
                    else:
                        fail_reasons["no_results.json"] += 1
                    continue

                with gzip.open(route_dir + '/results.json.gz', 'rt') as f:
                    total_routes += 1
                    try:
                        results_route = ujson.load(f)
                    except Exception as e:
                        print(f"Error in {route_dir}")
                        print(e)
                        if "results.json_load_error" not in fail_reasons:
                            fail_reasons["results.json_load_error"] = 1
                        else:
                            fail_reasons["results.json_load_error"] += 1
                        continue

                if results_route['scores']['score_composed'] < 100.0:  # we also count imperfect runs as failed (except minspeedinfractions)
                    cond1 = results_route['scores']['score_route'] > 94.0  # we allow 6% of the route score to be missing
                    cond2 = results_route['num_infractions'] == (len(results_route['infractions']['min_speed_infractions']) + len(results_route['infractions']['outside_route_lanes']))
                    if not (cond1 and cond2):  # if the only problem is minspeedinfractions, keep it
                        crashed_routes += 1
                        if "route_crashed" not in fail_reasons:
                            fail_reasons["route_crashed"] = 1
                        else:
                            fail_reasons["route_crashed"] += 1
                        continue

            perfect_routes += 1

            # if not os.path.exists(route_dir + f'/{self.rgb_folder}'):
            #     if "no_rgb_folder" not in fail_reasons:
            #         fail_reasons["no_rgb_folder"] = 1
            #     else:
            #         fail_reasons["no_rgb_folder"] += 1
            #     continue

            num_seq = len(os.listdir(route_dir + f'/{self.rgb_folder}'))

            for seq in range(self.skip_first_n_frames, num_seq - self.pred_len - self.hist_len - 1):
                image = []
                box = []
                measurement = []
                augment_exist = False

                measurement_file = route_dir + '/measurements' + f'/{(seq + self.hist_len-1):04}.json.gz'

                if evaluation and measurement_file not in self.all_eval_samples:
                    continue
                
                if dreamer:
                    dreamer_file_path = measurement_file.replace('measurements', f'{self.dreamer_folder}').replace('data/', f'{self.dreamer_folder}/')
                    if not os.path.exists(dreamer_file_path):
                        continue
                 
                if self.bucket_name is not None and self.bucket_name != "all":
                    measurement_file_path = Path(measurement_file)
                    if str(measurement_file_path.parent) in run_id_dict:
                        if measurement_file_path.name not in run_id_dict[str(measurement_file_path.parent)]:
                            if "measurement_file_not_in_bucket" not in fail_reasons:
                                fail_reasons["measurement_file_not_in_bucket"] = 1
                            else:
                                fail_reasons["measurement_file_not_in_bucket"] += 1
                            continue
                    else:
                        if "measurement_folder_not_in_bucket" not in fail_reasons:
                            fail_reasons["measurement_folder_not_in_bucket"] = 1
                        else:
                            fail_reasons["measurement_folder_not_in_bucket"] += 1
                        continue

                # Loads the current (and past) frames (if seq_len > 1)
                skip = False
                augment_exist = True
                for idx in range(self.hist_len):
                    image.append(route_dir +  f'/{self.rgb_folder}' + (f'/{(seq + idx):04}.jpg'))
                    box.append(route_dir + '/boxes' + (f'/{(seq + idx):04}.json.gz'))

                if skip:
                    if "file_not_found" not in fail_reasons:
                        fail_reasons["file_not_found"] = 1
                    else:
                        fail_reasons["file_not_found"] += 1
                    continue

                measurement.append(route_dir + '/measurements')

                self.images.append(image)
                self.boxes.append(box)
                self.measurements.append(measurement)
                self.sample_start.append(seq)
                self.augment_exists.append(augment_exist)
                if dreamer:
                    self.alternative_trajectories.append(dreamer_file_path)

        # There is a complex "memory leak"/performance issue when using Python
        # objects like lists in a Dataloader that is loaded with
        # multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects
        # because they only have 1 refcount.
        self.images = np.array(self.images).astype(np.string_)
        self.boxes = np.array(self.boxes).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        if dreamer:
            self.alternative_trajectories = np.array(self.alternative_trajectories).astype(np.string_)

        self.sample_start = np.array(self.sample_start)
        # if rank == 0:
        print(f'[{self.split} samples]: Loading {len(self.images)} images from {self.data_path} for bucket {self.bucket_name}')
        print('Total amount of routes:', total_routes)
        print('Crashed routes:', crashed_routes)
        print('Perfect routes:', perfect_routes)
        print('Fail reasons:', fail_reasons)

    def __len__(self):
        """Returns the length of the dataset. """
        return self.images.shape[0]
    

    def load_current_and_future_measurements(self, measurements, sample_start):
        loaded_measurements = []

        ######################################################
        ######## load current and future measurements ########
        ######################################################

        # Since we load measurements for future time steps, we load and store them separately
        for i in range(self.hist_len):
            measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

            with gzip.open(measurement_file, 'rt') as f1:
                measurements_i = ujson.load(f1)
            loaded_measurements.append(measurements_i)

        end = self.pred_len + self.hist_len
        start = self.hist_len

        for i in range(start, end):
            try:
                measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

                with gzip.open(measurement_file, 'rt') as f1:
                    measurements_i = ujson.load(f1)
                loaded_measurements.append(measurements_i)
            except FileNotFoundError:
                # If the file is not found, we just use the last available measurement
                print(f"File not found: {measurement_file}")
                loaded_measurements.append(loaded_measurements[-1])
        current_measurement = loaded_measurements[self.hist_len - 1]
        measurement_file_current = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + start-1):04}.json.gz')
        return loaded_measurements, current_measurement, measurement_file_current

    def load_waypoints(self, data, loaded_measurements, aug_translation=0.0, aug_rotation=0.0):

        waypoints = self.get_waypoints(loaded_measurements[self.hist_len - 1:],
                                                                        y_augmentation=aug_translation,
                                                                        yaw_augmentation=aug_rotation)
        data['waypoints'] = np.array(waypoints[1:-1])

        waypoints_org = self.get_waypoints(loaded_measurements[self.hist_len - 1:],
                                                                        y_augmentation=0,
                                                                        yaw_augmentation=0)
        data['waypoints_org'] = np.array(waypoints_org[1:-1])

        # 1D waypoints: only consider distance between waypoints
        waypoints_1d = [np.linalg.norm(waypoints_org[i+1] - waypoints_org[i]) for i in range(len(waypoints_org)-1)]
        # cumsum to get the distance from the start
        waypoints_1d = np.cumsum(waypoints_1d)
        waypoints_1d = [[x, 0] for x in waypoints_1d]
        data['waypoints_1d'] = np.array(waypoints_1d[:-1]).reshape(-1, 2)

        waypoints = [np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]) for x, y in waypoints]
        data['ego_waypoints'] = np.array(waypoints[:-1])
        
        waypoints_org = [np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]) for x, y in waypoints_org]
        data['ego_waypoints_org'] = np.array(waypoints_org[:-1])

        return data
    
    def load_route(self, data, current_measurement, aug_translation=0.0, aug_rotation=0.0):
        route = current_measurement['route_original']
        route = self.augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        route_adjusted = np.array(current_measurement['route'])
        route_adjusted_org = self.augment_route(route_adjusted, y_augmentation=0, yaw_augmentation=0)
        route_adjusted = self.augment_route(route_adjusted, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        if len(route) < self.num_route_points:
            num_missing = self.num_route_points - len(route)
            route = np.array(route)
            # Fill the empty spots by repeating the last point.
            route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
        else:
            route = np.array(route[:self.num_route_points])
            
        route_adjusted = self.equal_spacing_route(route_adjusted)
        route_adjusted_org = self.equal_spacing_route(route_adjusted_org)
        route = self.equal_spacing_route(route)
        
        data['route'] = route
        data['route_adjusted_org'] = route_adjusted_org
        data['route_adjusted'] = route_adjusted

        return data
    
    def load_images(self, data, images, augment_sample=False):
        loaded_images = []
        loaded_images_org_size = []
        for i in range(self.hist_len):
            images_i = None
            images_path = str(images[i], encoding='utf-8')
            if augment_sample:
                images_path = images_path.replace('rgb', 'rgb_augmented')

            if not os.path.isfile(images_path):
                print(f"File not found: {images_path}")
                raise FileNotFoundError

            images_i = cv2.imread(images_path, cv2.IMREAD_COLOR)
            images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)

            if self.img_augmentation: # and random.random() <= self.img_augmentation_prob:
                images_i = self.tfs(image=images_i)
            
            image_org = images_i.copy()
            if self.cut_bottom_quarter or self.img_shift_augmentation:
                # to remove the bonnet whih is important for the shifted camera augmentation
                # we need to remove 4.8/16 of the bottomf of the image (empirical value)
                images_i = images_i[:int(images_i.shape[0] - (images_i.shape[0] * 4.8) // 16), :, :]

            loaded_images.append(images_i)
            loaded_images_org_size.append(image_org)
        
        processed_image = np.asarray(loaded_images)
        processed_image_org_size = np.asarray(loaded_images_org_size)

        # we want [T, N, C, H, W], T is the number of temporal frames, N is the number of cam views, C is the number of channels, H is the height and W is the width
        processed_image = np.transpose(processed_image, (0, 3, 1, 2)) # (T, C, H, W)
        processed_image_org_size = np.transpose(processed_image_org_size, (0, 3, 1, 2)) # (T, C, H, W)

        data['rgb'] = processed_image
        data['rgb_org_size'] = processed_image_org_size

        return data

    def get_navigational_conditioning(self, data, current_measurement, target_point, next_target_point):
        placeholder_values = {}
        target_options = []
                
        tp = [target_point, next_target_point]
        tp = np.array(tp)
        data['map_route'] = tp
        data['target_points'] = tp
        target_point1_round = np.round(data['target_points'][0], 2).tolist()
        target_point2_round = np.round(data['target_points'][1], 2).tolist()

        if 'target_point' in self.route_as:
            if 'target_point_language' in self.route_as:
                target_options.append(f"Target waypoint: 1:{target_point1_round} 2:{target_point2_round}")
            else:
                target_options.append(f"Target waypoint: <TARGET_POINT><TARGET_POINT>.")
                placeholder_values = {'<TARGET_POINT>': data['target_points']}
        if 'command' in self.route_as:
            # get distance from target_point
            dist_to_command = np.linalg.norm(target_point)
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
            command = map_command[current_measurement["command"]]
            next_command = map_command[current_measurement["next_command"]]
            if command != next_command:
                next_command = f' then {next_command}'
            else:
                next_command = ''
            if current_measurement["command"] == 4:
                command_str = f'Command: {command}{next_command}.'
            else:
                command_str = f'Command: {command} in {dist_to_command} meter{next_command}.'
            target_options.append(command_str)
            
            if self.use_lmdrive_commands:
                lmdrive_index = random.choice(command_template_mappings[current_measurement["command"]])
                lmdrive_command = random.choice(self.command_templates[str(lmdrive_index)])
                lmdrive_command = lmdrive_command.replace('[x]', str(dist_to_command))
                lm_command = f'Command: {lmdrive_command}.'
                target_options.append(lm_command)
        
        return target_options, placeholder_values

    def equal_spacing_route(self, points):
        route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
        shift = np.roll(route, 1, axis=0) # Shift by 1
        shift[0] = shift[1] # Set wraparound value to 0

        dists = np.linalg.norm(route-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

        x = np.arange(0, 20, 1)
        interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

        return interp_points
    
    def visualise_cameras(
        self,
        batch: DatasetOutput,
        language, route, waypoints,
        options,
        name: str = "img",
        prompt=None,
        answer=None,
    ) -> np.ndarray:
        
        fov = 110

        img_front_np = batch.image_ff_org_size #[0, ...]
        img_front_np = img_front_np.transpose(0, 2, 3, 1)
        # two patches..dim 1 of img_front is 2 (left and right patches)
        # concatenate them to get a single image
        # img_front_1 = img_front[:, 0, ...]
        # img_front_2 = img_front[:, 1, ...]
        # img_front_torch = torch.cat((img_front_1, img_front_2), dim=3)

        all_images = [Image.fromarray((img_front_np[i])) for i in range(1)]

        # all_images = [Image.fromarray((img_front_torch[i].cpu().permute(1, 2, 0).numpy())) for i in range(1)]
        all_draws = [ImageDraw.Draw(image) for image in all_images]
        
        # black image to be concatenated to the bottom of the image
        img_width = all_images[0].size[0]
        text_box = [Image.new("RGB", (img_width, 200), "black") for _ in range(1)]
        text_draw = [ImageDraw.Draw(image) for image in text_box]
        
        W=all_images[0].size[0]
        H=all_images[0].size[1]
        camera_intrinsics = np.asarray(get_camera_intrinsics(W,H,fov))

        for i in range(1):
            gt_waypoints_img_coords = project_points(batch.waypoints, camera_intrinsics)
            for points_2d in gt_waypoints_img_coords:
                all_draws[i].ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(0, 255, 0, 255))

            if route is not None:
                pred_route_img_coords = project_points(route, camera_intrinsics)
                for points_2d in pred_route_img_coords:
                    all_draws[i].ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(255, 0, 0, 255))

            if language is not None:
                y_curr = 10
                
                # write the language to the bottom of the image
                # all_draws[i].rectangle([0, H-60, W, H], fill=(0, 0, 0, 255))
                # all_draws[i].text((10, H-40), f"Pred: {language[i]}", fill=(255, 255, 255, 255))
                text_draw[i].text((10, y_curr), f"Commentary: {language}", fill=(255, 255, 255, 255))
            if prompt is not None:
                text_draw[i].text((10, 30), f"Prompt: {prompt}", fill=(255, 255, 255, 255))
            if answer is not None:
                text_draw[i].text((10, 50), f"Answer: {answer}", fill=(255, 255, 255, 255))
                
        # concat text box to the bottom of the image
        
        # duplicate all_images len(all_negatives) times, deepcopy!
        if options is not None:
            all_all_images = [None for _ in range(len(options))]
            all_blacks = [None for _ in range(len(options))]
            for i, option in enumerate(options):
                wp_altern = option['waypoints']
                route_altern = option['route']
                if isinstance(route_altern, str) and route_altern == 'org':
                    route_altern = route
                if 'dreamer_instruction' in option:
                    language = option['dreamer_instruction'][0] if isinstance(option['dreamer_instruction'], list) else option['dreamer_instruction']
                    answer = option['dreamer_answer_safety'][0] if isinstance(option['dreamer_answer_safety'], list) else option['dreamer_answer_safety']
                else:
                    language = None
                    answer = None
                img = all_images[0].copy()
                draw = ImageDraw.Draw(img)
                img_black = text_box[0].copy()
                draw_black = ImageDraw.Draw(img_black)
                gt_waypoints_img_coords = project_points(wp_altern, camera_intrinsics)
                for points_2d in gt_waypoints_img_coords:
                    draw.ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(0, 55, 0, 255))
                if route_altern is not None:
                    pred_route_img_coords = project_points(route_altern, camera_intrinsics)
                    for points_2d in pred_route_img_coords:
                        draw.ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(55, 0, 0, 255))
                if language is not None:
                    draw_black.text((10, 80), f"Alternative Traj: {language}", fill=(255, 255, 255, 255))
                    draw_black.text((10, 100), f"Alternative Traj: {answer}", fill=(255, 255, 255, 255))
                    
                all_all_images[i] = img
                all_blacks[i] = img_black
            
        all_images = [Image.fromarray(np.concatenate([np.array(image), np.array(text)], axis=0)) for image, text in zip(all_images, text_box)]
        if options is not None:
            all_images.extend([Image.fromarray(np.concatenate([np.array(image), np.array(text)], axis=0)) for image, text in zip(all_all_images, all_blacks)])
        
        # concat all images
        viz_image_np = np.concatenate([np.array(image) for image in all_images], axis=0)
        viz_image = Image.fromarray(viz_image_np)
        
        # get ucrrent work dir
        current_dir = os.getcwd()
        
        Path("viz_images").mkdir(parents=True, exist_ok=True)
        # save the image
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_image.save(f"viz_images/{name}_{time}.png")

        return viz_image_np
    

    def get_indices_speed_angle(self, target_speed, brake):
        target_speeds = [0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20, 24]  # v4 target speeds (0.72*speed limits) plus extra classes for obstacle scenarios and intersecions

        target_speed_bins = [x+0.001 for x in target_speeds[1:]]  
        target_speed_index = np.digitize(x=target_speed, bins=target_speed_bins)

        # Define the first index to be the brake action
        if brake:
            target_speed_index = 0
        else:
            target_speed_index += 1

        return target_speed_index

    def augment_route(self, route, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                    np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0, y_augmentation]])
        route_aug = (rotation_matrix.T @ (route - translation).T).T
        return route_aug

    def augment_target_point(self, target_point, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0], [y_augmentation]])
        pos = np.expand_dims(target_point, axis=1)
        target_point_aug = rotation_matrix.T @ (pos - translation)
        return np.squeeze(target_point_aug)

    def parse_bounding_boxes(self, boxes, future_boxes=None, y_augmentation=0.0, yaw_augmentation=0):

        bboxes = []
        for current_box in boxes:
            # Ego car is always at the origin. We don't predict it.
            if current_box['class'] == 'ego_car':
                continue

            if 'position' not in current_box or 'extent' not in current_box:
                continue

            bbox, height = self.get_bbox_label(current_box, y_augmentation, yaw_augmentation)

            if current_box['class'] == 'traffic_light':
                # Only use/detect boxes that are red and affect the ego vehicle
                if not current_box['affects_ego']:
                    continue

            if current_box['class'] == 'stop_sign':
                # Don't detect cleared stop signs.
                if not current_box['affects_ego']:
                    continue

            bboxes.append(bbox)
        return bboxes

    def get_bbox_label(self, bbox_dict, y_augmentation=0.0, yaw_augmentation=0):
        # augmentation
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                np.cos(aug_yaw_rad)]])

        position = np.array([[bbox_dict['position'][0]], [bbox_dict['position'][1]]])
        translation = np.array([[0.0], [y_augmentation]])

        position_aug = rotation_matrix.T @ (position - translation)

        x, y = position_aug[:2, 0]
        # center_x, center_y, w, h, yaw
        bbox = np.array([x, y, bbox_dict['extent'][0], bbox_dict['extent'][1], 0, 0, 0, 0, 0])
        bbox[4] = t_u.normalize_angle(bbox_dict['yaw'] - aug_yaw_rad)

        if bbox_dict['class'] == 'car':
            bbox[5] = bbox_dict['speed']
            bbox[6] = bbox_dict['brake']
            bbox[7] = 0
        elif bbox_dict['class'] == 'walker':
            bbox[5] = bbox_dict['speed']
            bbox[7] = 1
        elif bbox_dict['class'] == 'traffic_light':
            bbox[7] = 2
            if bbox_dict['state'] == 'Green':
                bbox[8] = 0
            elif bbox_dict['state'] == 'Red' or bbox_dict['state'] == 'Yellow':
                bbox[8] = 1
            else:
                bbox[8] = 2
        elif bbox_dict['class'] == 'stop_sign':
            bbox[7] = 3

        else:
            bbox = np.zeros(9)
        return bbox, bbox_dict['position'][2]

    def get_route_image(self, route, target_point):
        route_img = np.zeros((64, 64, 3), dtype=np.uint8)
        route_new = np.array(route, dtype=np.float32)
        route_new[:, 0] = -route_new[:, 0]*2 + 63
        route_new[:, 1] = route_new[:, 1]*2 + 32
        route_new = route_new.clip(0, 63)
        route_new = route_new.astype(np.int32)
        route_img[route_new[:, 0], route_new[:, 1], :] = 255

        # # target point as red
        # target_point = np.array(target_point, dtype=np.float32)
        # target_point[0] = -target_point[0]*2 + 63
        # target_point[1] = target_point[1]*2 + 32
        # target_point = target_point.astype(np.int32)
        # # target_point = target_point.clip(0, 63)
        # route_img[target_point[0], target_point[1], 0] = 255

        # save route_img
        # cv2.imwrite('/home/wayve/katrinrenz/coding/WayveCode/route_img.png', route_img)

        return route_img

    def get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
        """transform waypoints to be origin at ego_matrix"""
        origin = measurements[0]
        origin_matrix = np.array(origin['ego_matrix'])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]

        waypoints = []
        for index in range(len(measurements)):
            waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
            waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
            # Drop the height dimension because we predict waypoints in BEV
            waypoints.append(waypoint_ego_frame[:2, 0])

        # Data augmentation
        waypoints_aug = []
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                                                                                            np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0], [y_augmentation]])
        for waypoint in waypoints:
            pos = np.expand_dims(waypoint, axis=1)
            waypoint_aug = rotation_matrix.T @ (pos - translation)
            waypoints_aug.append(np.squeeze(waypoint_aug))

        return waypoints_aug

def image_augmenter(prob=0.2, cutout=False):
    augmentations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
        ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=0.5)),
        ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),  # Strong
        ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
    ]

    if cutout:
        augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))

    augmenter = ia.Sequential(augmentations, random_order=True)

    return augmenter


def get_camera_intrinsics(w, h, fov):
    """
    Get camera intrinsics matrix from width, height and fov.
    Returns:
        K: A float32 tensor of shape ``[3, 3]`` containing the intrinsic calibration matrices for
            the carla camera.
    """

    # print(f"[CAMERA MATRIX] Load camera intrinsics for TF++ default camera with w: {w}, h: {h}, fov: {fov}")

    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    K = torch.tensor(K, dtype=torch.float32)
    return K

def get_camera_extrinsics():
    """
    Get camera extrinsics matrix for the carla camera.
    extrinsics: A float32 tensor of shape ``[4, 4]`` containing the extrinic calibration matrix for
            the carla camera. The extriniscs are specified as homogeneous matrices of the form ``[R t; 0 1]``
    """

    # camera_pos = [-1.5, 0.0, 2.0]  # x, y, z mounting position of the camera
    # camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

    # print("[CAMERA MATRIX] Load camera extrinsics for TF++ default camera with x: -1.5, y: 0.0, z: 2.0, roll: 0.0, pitch: 0.0, yaw: 0.0")
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    extrinsics[3, 3] = 1.0
    extrinsics[:3, :3] = np.eye(3)
    extrinsics[:3, 3] = [-1.5, 0.0, 2.0]

    extrinsics = torch.tensor(extrinsics, dtype=torch.float32)

    return extrinsics

def get_camera_distortion():
    """
    Get camera distortion matrix for the carla camera.
    distortion: A float32 tensor of shape ``[14 + 1]`` containing the camera distortion co-efficients
            ``[k0, k1, ..., k13, d]`` where ``k0`` to ``k13`` are distortion co-efficients and d specifies the
            distortion model as defined by the DistortionType enum in camera_info.hpp
    """

    print("[CAMERA MATRIX] Load camera distortion for TF++ default camera. No distortion.")
    distortion = np.zeros(14 + 1, dtype=np.float32)
    distortion[-1] = 0.0
    distortion = torch.tensor(distortion, dtype=torch.float32)

    return distortion




if __name__ == "__main__":
    from hydra import compose, initialize
    from simlingo_training.config import TrainConfig
    
    # set all seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    

    initialize(config_path="../config")
    cfg = compose(config_name="config")
    
    cfg.data_module.base_dataset.use_commentary = False
    cfg.data_module.base_dataset.img_shift_augmentation = True
    
    cfg.data_module.base_dataset.use_safety_flag = True

    print('Test Dataset')
    dataset = Data_Dreamer(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(data)
        if i == 100:
            break