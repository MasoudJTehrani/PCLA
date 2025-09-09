# Standard library imports
import itertools
from typing import List

# Third-party imports
import hydra
import line_profiler
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# Local/project specific imports
# from simlingo_training.dataloader.dataset_driving import Data_Driving # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
# from simlingo_training.dataloader.dataset_dreamer import Data_Dreamer # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
from simlingo_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel, LanguageLabel
from simlingo_training.utils.internvl2_utils import preprocess_image_batch, get_custom_chat_template, get_num_image_tokens_per_patch
from simlingo_training.utils.projection import get_camera_intrinsics, get_camera_extrinsics

def encode_uint8(strings: List[str], common_length: int) -> torch.Tensor:
    max_len = max(len(s) for s in strings)
    assert max_len <= common_length, f"String is too long: {max_len} > {common_length}"
    padded_strings = [s.ljust(common_length, '\0') for s in strings]
    return torch.tensor([bytearray(s, 'utf-8') for s in padded_strings], dtype=torch.uint8)


class DataModule(LightningDataModule):
    def __init__(
        self,
        base_dataset,
        processor,
        predict=False,
        **cfg,
    ):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
            
        for key, value in base_dataset.items():
            setattr(self, key, value)
            
        self.cfg = cfg
        self.base_dataset = base_dataset
        self.processor = processor
        self.predict = predict
        
        self.printed = False

        self.NUM_IMAGE_PATCHES = 2
        self.IMAGES_TO_CONSIDER = ['image_ff'] # front-forward image, other images are not supported
        # taken from:
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/train/constants.py#L7
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L294
        self.IMG_START_TOKEN='<img>'
        self.IMG_END_TOKEN='</img>'
        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        self.num_image_tokens_per_patch = get_num_image_tokens_per_patch(self.encoder_variant)
        self.num_image_tokens_total = self.num_image_tokens_per_patch * self.NUM_IMAGE_PATCHES
            
        # add <WAYPOINT> token
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
        # TODO: not needed anymore?
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"

    def setup(self, stage=None):
        if not self.predict:
            self.val_datasets = []
            sum_sample_weights = 1.0
            bucket_list = []
            num_datasets = 0
            datasets = {}
            sample_weights = []
            
            if self.driving_dataset is not None or self.dreamer_dataset is not None:
                # Create lists of datasets and their corresponding training partitions
                used_driving_datasets = []
                used_train_partitions = []
                
                # Pair datasets with their training partitions and filter out None datasets
                dataset_pairs = zip(
                    [self.driving_dataset, self.dreamer_dataset],
                    [self.train_partitions, self.train_partitions_dreamer]
                )
                
                for dataset, partition in dataset_pairs:
                    if dataset is not None:
                        used_driving_datasets.append(dataset)
                        used_train_partitions.append(partition)
                
                weights_driving = 0.5
                weights_dreamer = 1 - weights_driving
                for udd_i, (used_driving_dataset, used_train_partitions) in enumerate(zip(used_driving_datasets, used_train_partitions)):
                    num_datasets += 1
                    if used_train_partitions is not None:
                        bucket_list_tmp = list(used_train_partitions.keys())
                        sample_weights_tmp = list(used_train_partitions.values())
                        sum_sample_weights = sum(sample_weights_tmp)
                            
                        sample_weights_tmp = [w/sum_sample_weights for w in sample_weights_tmp]
                        if self.driving_dataset is not None and self.dreamer_dataset is not None:

                            if udd_i == 0:
                                sample_weights_tmp = [w * weights_driving for w in sample_weights_tmp]
                            else:
                                sample_weights_tmp = [w * weights_dreamer for w in sample_weights_tmp]

                        if udd_i == 1:
                            bucket_list_tmp = [f"{b}_dreamer" for b in bucket_list_tmp]
                        bucket_list.extend(bucket_list_tmp)
                        sample_weights.extend(sample_weights_tmp)
                    else:
                        if udd_i == 1:
                            bucket_list_tmp = ['all_dreamer']
                            sample_weights_tmp = [weights_dreamer]
                        else:
                            bucket_list_tmp = ['all']
                            sample_weights_tmp = [weights_driving]
                        bucket_list.extend(bucket_list_tmp)
                        sample_weights.extend(sample_weights_tmp)
                    

                    for bucket in bucket_list_tmp:
                        bucket_name = bucket.replace('_dreamer','')
                        datasets[bucket] = hydra.utils.instantiate(
                            used_driving_dataset,
                            split="train",
                            bucket_name=bucket_name,
                            **self.cfg,
                            **self.base_dataset,
                            _recursive_=False
                        )
                    self.val_datasets.append(hydra.utils.instantiate(
                            used_driving_dataset,
                            split="val",
                            bucket_name="all",
                            **self.cfg,
                            **self.base_dataset,
                            _recursive_=False
                        ))
                    
                    sum_sample_weights = sum(sample_weights_tmp)
            

            
            self.train_dataset = None
            if len(datasets) > 0:
                
                # remove datasets with 0 samples
                sample_weights = [sample_weights[i] for i, bucket in enumerate(bucket_list) if datasets[bucket].__len__() > 0]
                bucket_list = [bucket for bucket in bucket_list if datasets[bucket].__len__() > 0]
                if len(bucket_list) != len(datasets):
                    # print in red
                    print(f"\033[91mDatasets with 0 samples: {set(datasets.keys()) - set(bucket_list)}\033[00m")
                    print(f"\033[91mContinue without this bucket.\033[00m")
                datasets = {key: value for key, value in datasets.items() if value.__len__() > 0}

                self.train_dataset = torch.utils.data.ConcatDataset([datasets[bucket] for bucket in bucket_list])
                weights_train = [[sample_weights[i]] * datasets[bucket].__len__() for i, bucket in enumerate(bucket_list)]
                weights_train = list(itertools.chain.from_iterable(weights_train))
                num_samples_all = [datasets[bucket].__len__() // sample_weights[i] for i, bucket in enumerate(bucket_list)]
                num_samples = int(min(num_samples_all))# * num_datasets
                print(f"Num samples: {num_samples}")
                if self.driving_dataset is not None:
                    print(f"Num samples all: {datasets['all'].__len__()}")
                self.sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=num_samples, replacement=True)

            self.val_dataset = torch.utils.data.ConcatDataset(self.val_datasets)
            self.predict_dataset = None

        else:
            if self.qa_dataset is not None:
                predict_dataset = self.qa_dataset
                
            elif self.insteval_dataset is not None:
                predict_dataset = self.insteval_dataset

            self.predict_dataset = hydra.utils.instantiate(
                    predict_dataset,
                    split="val",
                    bucket_name="all",
                    **self.cfg,
                    **self.base_dataset,
                    _recursive_=False
                )


    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # shuffle=True, # we use custom sampler instead
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            sampler=self.sampler_train,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )


    @line_profiler.profile
    def dl_collate_fn(self, data):
        BS = len(data)
        grid_nums = [self.NUM_IMAGE_PATCHES] # we split the front forward into two patches (1x2)

        image_ff_pixel, image_ff_sizes = None, None
        image_ff_org = torch.tensor(np.asarray([data[i].image_ff_org_size for i in range(BS)]))
            
        for idx, img_to_consider in enumerate(self.IMAGES_TO_CONSIDER):
            img_tmp = getattr(data[0], img_to_consider)
            T, C, H, W = img_tmp.shape
            assert T == 1, "Only one timestep as input supported"
            
            images_batch_tensor = torch.tensor(np.asarray([getattr(data[i], img_to_consider) if getattr(data[i], img_to_consider) is not None else np.zeros_like(img_tmp) for i in range(len(data))])).float()
            images_batch_tensor = images_batch_tensor.view(BS*T, C, H, W)
            images_batch_list = list(images_batch_tensor)

            if 'internvl2' in self.encoder_variant.lower():
                # get image patches
                images_processed = preprocess_image_batch(images_batch_list, input_size=448, use_global_img=self.use_global_img, max_num_grid=grid_nums[idx])    
            else:
                raise ValueError(f"Image preprocessing for {self.encoder_variant} not implemented")
                
            images_pixel = images_processed['pixel_values']
            image_sizes = images_processed['image_sizes']
            
            assert images_pixel.shape[0] == BS * T
            num_patches = images_pixel.shape[1]
            assert images_pixel.shape[2] == C
            new_height = images_pixel.shape[3]
            new_width = images_pixel.shape[4]
            images_pixel = images_pixel.view(BS, T, num_patches, C, new_height, new_width)
            
            if img_to_consider == 'image_ff':
                image_ff_pixel = images_pixel
                image_ff_sizes = image_sizes
            else:
                raise ValueError(f"Image type {img_to_consider} not supported")

        conversations = [data[i].conversation for i in range(BS)]
        conversation_dict, question_dict = get_custom_chat_template(conversations, self.tokenizer, self.encoder_variant, self.num_image_tokens_total)

        placeholder_batch_list = []
        for i in range(BS):
            tmp = {}
            for key, value in data[i].placeholder_values.items():
                token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)
                
        prompt_languagelabel = LanguageLabel(
            phrase_ids=conversation_dict['phrase_ids'],
            phrase_valid=conversation_dict['phrase_valid'],
            phrase_mask=conversation_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=conversation_dict['language_string'],
            loss_masking=conversation_dict['loss_masking'],
        )

        prompt_question_languagelabel = LanguageLabel(
            phrase_ids=question_dict['phrase_ids'],
            phrase_valid=question_dict['phrase_valid'],
            phrase_mask=question_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=question_dict['language_string'],
            loss_masking=question_dict['loss_masking'],
        )
        answer_string_list = [data[i].answer[0]['content'][0]['text'] for i in range(BS)]
        answer_label =  LanguageLabel(
            phrase_ids=None,
            phrase_valid=None,
            phrase_mask=None,
            placeholder_values=None,
            language_string=answer_string_list,
            loss_masking=None,
        )
        
        if self.base_dataset.use_1d_wps:
            waypoints = torch.tensor(np.asarray([data[i].waypoints_1d for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        else:
            waypoints = torch.tensor(np.asarray([data[i].waypoints for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        
        if self.predict:
            qa_templates = [data[i].qa_templates[0] if data[i].qa_templates is not None else None for i in range(BS) ]
            eval_infos = [data[i].eval_infos if data[i].eval_infos is not None else None for i in range(BS) ]
        else:
            qa_templates = None
            eval_infos = None
        
        driving_input=DrivingInput(
                camera_images=image_ff_pixel,  # [B, T, N, C, H, W] uint8 [0, 255]
                image_sizes=image_ff_sizes,
                camera_intrinsics = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), BS, dim=0).view(BS, 3, 3).float(),
                camera_extrinsics = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), BS, dim=0).view(BS, 4, 4).float(),
                vehicle_speed=torch.tensor(np.asarray([data[i].speed for i in range(len(data))])).float(),  # [B, S] float32
                target_point=torch.tensor(np.asarray([data[i].target_points for i in range(len(data))])).float(),  # [B, 2] float32
                prompt=prompt_languagelabel,
                prompt_inference=prompt_question_languagelabel,
            )

        driving_label=DrivingLabel(
                waypoints=waypoints,
                path=torch.tensor(np.asarray([data[i].path for i in range(len(data))])).float(), # [B, 3, RH, RW] uint8 [0, 255]
                answer=answer_label,
                image_ff_org=image_ff_org,
                eval_infos=eval_infos,
            )
            
        return DrivingExample(
            driving_input=driving_input,
            driving_label=driving_label,
            run_id=encode_uint8([data[i].measurement_path for i in range(BS)], 1000),  # [B] str
            qa_templates=qa_templates,
        )

    def dl_collate_fn_val(self, data):
        pass

    def dl_collate_fn_test(self, data):
        pass


@hydra.main(config_path=f"../config", config_name="config", version_base="1.1")
def test(cfg):
    
    get_waypoint_stats = True
    
        
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    dm = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        # tokenizer=llm_tokenizer,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant="llava-hf/llava-v1.6-mistral-7b-hf",
        _recursive_=False
    )


    dm.setup()
    dl = dm.val_dataloader()
    print(dl.dataset.__len__())

    iterations = 0
    
    all_waypoints = []
    all_waypoints_diff = []
    for batch in dl:
        
        iterations += 1
        
        if iterations % 100 == 0:
            print(f"Iteration: {iterations}")
        
        if iterations > 20000:
            break
        
        if get_waypoint_stats:
            # get stats about range of waypoints
            waypoints = batch.driving_label.waypoints
            all_waypoints.append(waypoints)
            
            # get residuals
            residuals = waypoints[:,1:] - waypoints[:,:-1]
            all_waypoints_diff.append(residuals)
            
    # get histogram of waypoints
    if get_waypoint_stats:
        all_waypoints = torch.cat(all_waypoints, dim=0)
        all_waypoints_diff = torch.cat(all_waypoints_diff, dim=0)
        
        all_waypoints = all_waypoints.view(-1, 2)
        all_waypoints_diff = all_waypoints_diff.view(-1, 2)
        
        
        
        import matplotlib.pyplot as plt
        plt.hist(all_waypoints[:,0].numpy(), bins=100)
        plt.savefig('waypoints_x.png')
        max_x = all_waypoints[:,0].max().item()
        min_x = all_waypoints[:,0].min().item()
        print(f"Max x: {max_x}, Min x: {min_x}")
        plt.clf()
        plt.hist(all_waypoints[:,1].numpy(), bins=100)
        plt.savefig('waypoints_y.png')
        max_y = all_waypoints[:,1].max().item()
        min_y = all_waypoints[:,1].min().item()
        print(f"Max y: {max_y}, Min y: {min_y}")
        plt.clf()
        
        plt.hist(all_waypoints_diff[:,0].numpy(), bins=100)
        plt.savefig('waypoints_diff_x.png')
        max_x_diff = all_waypoints_diff[:,0].max().item()
        min_x_diff = all_waypoints_diff[:,0].min().item()
        print(f"Max x diff: {max_x_diff}, Min x diff: {min_x_diff}")
        plt.clf()
        plt.hist(all_waypoints_diff[:,1].numpy(), bins=100)
        plt.savefig('waypoints_diff_y.png')
        max_y_diff = all_waypoints_diff[:,1].max().item()
        min_y_diff = all_waypoints_diff[:,1].min().item()
        print(f"Max y diff: {max_y_diff}, Min y diff: {min_y_diff}")
        plt.clf()
            
            

if __name__ == "__main__":
    test()