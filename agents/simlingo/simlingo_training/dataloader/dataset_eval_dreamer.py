"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
import numpy as np
import random
import cv2
import gzip

import torch
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.dataloader.dataset_base import BaseDataset


VIZ_DATA = False

class Eval_Dreamer(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(dreamer=True, **cfg)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]
        alternative_trajectories = self.alternative_trajectories[index]

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current

        if self.use_safety_flag:
            if random.random() < 0.5:
                activate_safety = True
            else:
                activate_safety = False
        else:
            activate_safety = None

        augment_sample = False
        aug_rotation = 0.0
        aug_translation = 0.0


        ######################################################
        ################## load waypoints ####################
        ######################################################
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       
        speed_rounded = round(current_measurement['speed'], 1)
        data['speed'] = current_measurement['speed']

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        ######################################################
        ################## get alternatives ##################
        ######################################################
        alternative_file = str(alternative_trajectories, encoding='utf-8')
        with gzip.open(alternative_file, 'rt') as f1:
            alternative_trajectories = ujson.load(f1)

        options = []
        for key, option in alternative_trajectories.items():
            if 'factor' in key:
                continue
            
            options.extend(option)

        chosen_option = random.choice(options)

        # replace 'org' with the original route
        if chosen_option['route'] == 'org':
            chosen_option['route'] = data['route_adjusted_org']
        else:
            chosen_option['route'] = np.array(chosen_option['route'])
        
        if chosen_option['waypoints'] == 'org':
            chosen_option['waypoints'] = data['waypoints_org']
        else:
            chosen_option['waypoints'] = np.array(chosen_option['waypoints'])
        
        chosen_option['dreamer_instruction'] = random.choice(chosen_option['dreamer_instruction'])

        dreamer_answer = f"Following the given instruction. Waypoints:"
        if activate_safety is not None:
            if activate_safety:
                if chosen_option['safe_to_execute']:
                    augment_sample = False
                else:
                    dreamer_answer = chosen_option['dreamer_answer_safety']
            else:
                augment_sample = False
        

        ######################################################
        ######## load navigational_conditioning ########
        ######################################################
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
            
        answer = ''

        if random.random() < 0.8:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {chosen_option['dreamer_instruction']}"
        else:
            prompt = f"Current speed: {speed_rounded} m/s. {chosen_option['dreamer_instruction']}"
            
        waypoints = chosen_option['waypoints']
        waypoints = np.array(waypoints)
        
        waypoints_zero = np.concatenate((np.zeros((1, 2)), waypoints), axis=0)
        waypoints_1d = [np.linalg.norm(waypoints_zero[i+1] - waypoints_zero[i]) for i in range(len(waypoints_zero)-1)]
        waypoints_1d = np.cumsum(waypoints_1d)
        waypoints_1d = [[x, 0] for x in waypoints_1d]
        waypoints_1d = np.array(waypoints_1d).reshape(-1, 2)
        
        path = chosen_option['route']
        answer = dreamer_answer

        prompt = prompt.replace('..', '.').replace('  ', ' ').replace('!.', '!').replace('?.', '?')
                
        ######################################################
        ######## load current and past images ########
        ######################################################
        data = self.load_images(data, images, augment_sample=augment_sample)
        
        # overwrite action when safety flag is active and action is not allowed
        if activate_safety is not None:
            if activate_safety:
                prompt = f"<SAFETY> {prompt}"
                if chosen_option['safe_to_execute'] == False:
                    waypoints = data['waypoints_org']
                    waypoints_1d = data["waypoints_1d"]
                    path = data['route_adjusted_org']
            else:
                prompt = f"<INSTRUCTION_FOLLOWING> {prompt}"

        conversation_answer = [
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            },
        ]
        conversation_all = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            }
        ]
        
        images = [data['rgb']]

        eval_infos = {
            'mode': chosen_option['mode'],
            'allowed': chosen_option['allowed'],
            'org_wps': data['waypoints_org'],
            'org_wps_1d': data["waypoints_1d"],
            'org_path': data['route_adjusted_org'],
            'new_wps': chosen_option['waypoints'],
            'new_path': chosen_option['route'],
        }

        data_new = DatasetOutput(
            conversation = conversation_all,
            answer = conversation_answer,
            image_ff = data['rgb'],
            image_ff_org_size=data['rgb_org_size'],
            waypoints = waypoints,
            waypoints_1d = waypoints_1d,
            path = path,
            target_points = data['target_points'],
            speed = data['speed'],
            placeholder_values = placeholder_values,
            measurement_path = data['measurement_path'],
            dataset = 'driving',
            eval_infos = eval_infos,
        )
        
        if VIZ_DATA:
            # front image with path and waypoints and commentary
            self.visualise_cameras(data_new, None, path, waypoints, options, name="dreamer_", prompt=prompt, answer=answer)
        return data_new