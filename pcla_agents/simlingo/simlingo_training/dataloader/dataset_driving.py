"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
import json
import numpy as np
import random
import cv2
import re
import gzip

import torch
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.dataloader.dataset_base import BaseDataset


VIZ_DATA = False

class Data_Driving(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(dreamer=False, **cfg)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current

        # Determine whether the augmented camera or the normal camera is used.
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
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
        ################## get commentary & qa ##################
        ######################################################
        commentary_exists = False
        commentary = ''
        if self.use_commentary:
            commentary_file_path = measurement_file_current.replace('measurements', 'commentary').replace('data/', 'commentary/') # TODO: move to config
            # do not use evaluation routes!!!
            if 'validation_' in commentary_file_path:
                commentary_exists = False
            else:
                try:
                    with gzip.open(commentary_file_path, 'rt') as f:
                        commentary_file = ujson.load(f)
                        commentary_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    commentary_exists = False
                    commentary_file = None

                if commentary_file is not None:

                    commentary = commentary_file['commentary']
                    # we only augment in 60% of the cases and use the default commentary in 40% of the cases
                    # augmentation is used to increase generalization to a broader set of sentences
                    # but we do not want to overfit to the augmented sentences
                    if self.commentary_augmentation and random.random() < 0.6:
                        if commentary_file['commentary_template'] in self.templates_commentary:
                            commentary = random.choice(self.templates_commentary[commentary_file['commentary_template']])
                            for key, value in commentary_file['placeholder'].items():
                                if key in commentary:
                                    commentary = commentary.replace(key, value)
                            # regex check if <OBJECT> or <LOCATION> or any other <> is still in commentary, if so use default commentary_file['commentary']
                            if re.search(r'<.*?>', commentary):
                                print(f"WARNING: {commentary} contains placeholders that are not replaced. Using default commentary.")
                                commentary = commentary_file['commentary']

                    commentary = commentary.replace('..', '.')
                    commentary = commentary.replace('in in', 'in')
        
        qa_exists = False
        if self.use_qa:
            qa_path = measurement_file_current.replace('measurements', 'vqa').replace('data/', 'drivelm/')
            if 'validation_' in qa_path:
                qa_exists = False
            else:
                try:
                    with gzip.open(qa_path, 'rt') as f:
                        qa = ujson.load(f)
                    qa_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    qa_exists = False
                    qa = None

            if qa_exists:
                qas = qa['QA']
                qas = [values for values in qas.values()] # list of lists
                qas = [item for sublist in qas for item in sublist] # flatten list
                while True:
                    qa_chosen = random.choice(qas)
                    qa_question = qa_chosen['Q']
                    qa_answer = qa_chosen['A']
                    
                    # TODO: make this nicer!!!
                    if 'There are no pedestrians.' in qa_answer or \
                            'There is no traffic light' in qa_answer or \
                            'There are no pedestrians.' in qa_answer or \
                            'No, the ego vehicle is not affected by a stop sign.' in qa_answer or \
                            'No, the ego vehicle is not affected by a junction.' in qa_answer or \
                            'There is no traffic light affecting the ego vehicle.' in qa_answer or \
                            'There is no stop sign affecting the ego vehicle.' in qa_answer or \
                            'There is no junction affecting the ego vehicle.' in qa_answer or \
                            'It is not possible to tell' in qa_answer or \
                            'There is no reason for the ego vehicle to brake.' in qa_answer:
                        # only keep in 20% of the cases
                        if random.random() < 0.2:
                            break
                    else:
                        break
                
                # we only augment in 60% of the cases and use the default QA in 40% of the cases
                # augmentation is used to increase generalization to a broader set of sentences
                # but we do not want to overfit to the augmented sentences
                if self.qa_augmentation and random.random() < 0.6:
                    qa_question_org = qa_question
                    qa_answer_org = qa_answer
                    locations = [
                        'nearby to the front of the ego vehicle',
                        'nearby to the front right of the ego vehicle',
                        'nearby to the front left of the ego vehicle',
                        'nearby on the left side of the ego vehicle',
                        'far to the front left of the ego vehicle',
                        'far to the front right of the ego vehicle',
                        'far to the front of the ego vehicle',
                        'far to the left side of the ego vehicle',
                        'far to the right side of the ego vehicle',
                        'to the front of the ego vehicle',
                        'to the front right of the ego vehicle',
                        'to the front left of the ego vehicle',
                        'on the left side of the ego vehicle',
                        'on the right side of the ego vehicle',
                    ]
                    objects = [value['Visual_description'] for key, value in qa['key_object_infos'].items()]
                    q_objects = []
                    a_objects = []
                    for object_type in objects:
                        if object_type in qa_question:
                            qa_question = qa_question.replace(object_type, '<OBJECT>')
                            q_objects.append(object_type)
                        if object_type in qa_answer:
                            qa_answer = qa_answer.replace(object_type, '<OBJECT>')
                            a_objects.append(object_type)
                    
                    q_location = ''
                    a_location = ''
                    for location in locations:
                        if location in qa_question:
                            qa_question = qa_question.replace(location, '<LOCATION>')
                            q_location = location
                        if location in qa_answer:
                            qa_answer = qa_answer.replace(location, '<LOCATION>')
                            a_location = location
                        
                    q_distance = re.search(r'in (\d+) m', qa_question)
                    qa_question = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_question)
                    a_distance = re.search(r'in (\d+) m', qa_answer)
                    qa_answer = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_answer)
                    if len(q_objects)==0:
                        q_objects = ['']
                    if len(a_objects)==0:
                        a_objects = ['']
                    
                    # in 40% of the cases we do not augment the question
                    if len(q_objects) > 1 or len(a_objects) > 1 or random.random() < 0.4: 
                        qa_question = qa_question_org
                        qa_answer = qa_answer_org
                    else:
                        if qa_question in self.q_augment:
                            qa_question = random.choice(self.q_augment[qa_question]).replace('<OBJECT>', q_objects[0]).replace('<LOCATION>', q_location)
                            if q_distance:
                                qa_question = qa_question.replace('<DISTANCE>', q_distance.group(1))
                        else:
                            print(f"WARNING: {qa_question} not in q_augment. Using default question.")
                            qa_question = qa_question_org
                        if qa_answer in self.a_augment:
                            qa_answer = random.choice(self.a_augment[qa_answer]).replace('<OBJECT>', a_objects[0]).replace('<LOCATION>', a_location)
                            if a_distance:
                                qa_answer = qa_answer.replace('<DISTANCE>', a_distance.group(1))
                        else:
                            print(f"WARNING: {qa_answer} not in a_augment. Using default answer.")
                            qa_answer = qa_answer_org

        ######################################################
        ######## load navigational_conditioning ########
        ######################################################
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
            
        answer = ''

        prompt_random = random.random()
        
        if self.use_commentary and commentary_exists and prompt_random < self.prompt_probabilities['commentary']:
            if random.random() < 0.2: # 20% of the time we give commentary as prompt
                if random.random() < 0.5:
                    prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {commentary} Predict the waypoints."
                else:
                    prompt = f"Current speed: {speed_rounded} m/s. Command: {commentary} Predict the waypoints."
                answer = f"Waypoints:"
            else:
                # 80% of the time we want to predict commentary
                prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} What should the ego do next?"
                answer = f"{commentary} Waypoints:"
            self.num_sampled_per_type['commentary'] += 1
            
        elif self.use_qa and qa_exists and prompt_random < (self.prompt_probabilities['qa'] + self.prompt_probabilities['commentary']):
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Q: {qa_question}"
            answer = f"A: {qa_answer}"
            self.num_sampled_per_type['qa'] += 1
            
        else:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Predict the waypoints."
            answer = f"Waypoints:"
            self.num_sampled_per_type['driving'] += 1

        # recalculate the probabilties after warmup (when more than 1000 samples have been sampled)
        # we do this in case we dont have qa or commentary for every sample otherwise it would lead to undersampling one of those
        if sum(self.num_sampled_per_type.values()) > 10000 and sum(self.num_sampled_per_type.values()) % 10000 == 0:
            self.prompt_probabilities = {key: 1/value for key, value in self.num_sampled_per_type.items()}
            self.prompt_probabilities = {key: value/sum(self.prompt_probabilities.values()) for key, value in self.prompt_probabilities.items()}
            print(f"Prompt probabilities: {self.prompt_probabilities}")
            print(f"Number of samples per type: {self.num_sampled_per_type}")
            
        answer = answer.replace('..', '.')
        prompt = prompt.replace('..', '.')

        ######################################################
        ######## load current and past images ########
        ######################################################
        data = self.load_images(data, images, augment_sample=augment_sample)
        

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

        data_new = DatasetOutput(
            conversation = conversation_all,
            answer = conversation_answer,
            image_ff = data['rgb'],
            image_ff_org_size=data['rgb_org_size'],
            waypoints = data["waypoints"],
            waypoints_1d = data["waypoints_1d"],
            path = data['route_adjusted'],
            target_points = data['target_points'],
            speed = data['speed'],
            placeholder_values = placeholder_values,
            measurement_path = data['measurement_path'],
            dataset = 'driving',
        )
        
        if VIZ_DATA:
            # front image with path and waypoints and commentary
            self.visualise_cameras(data_new, commentary, data['route_adjusted'], data['waypoints'], options=None, prompt=prompt, answer=answer, name="img")
        return data_new


if __name__ == "__main__":
    from hydra import compose, initialize
    from simlingo_training.config import TrainConfig
    
    # seed all
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

    initialize(config_path="../config")
    cfg = compose(config_name="config")
    
    cfg.data_module.base_dataset.use_commentary = True
    cfg.data_module.base_dataset.use_qa = True
    cfg.data_module.base_dataset.img_shift_augmentation = False

    print('Test Dataset')
    dataset = Data_Driving(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        # shuffle
        # i = np.random.randint(0, len(dataset))
        data = dataset[i]
        # print(data)
        # if i == 100:
        #     break