import datetime
import json
import os
import random
from pathlib import Path
from pprint import PrettyPrinter
from typing import Dict, Optional, Tuple, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from hydra.utils import get_original_cwd


from simlingo_training.models.adaptors.adaptors import DrivingAdaptor, LanguageAdaptor, WaypointInputAdaptor, AdaptorList
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import (DrivingExample, DrivingInput,
                                                DrivingLabel, DrivingOutput,
                                                TrainingOutput)


pprint = PrettyPrinter().pprint

def decode_uint8(encoded: torch.Tensor) -> List[str]:
    return [row.tobytes().decode("utf-8").rstrip("\0") for row in encoded.cpu().numpy()]

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])


class DrivingModel(pl.LightningModule):
    def __init__(
        self,
        cfg_data_module,
        processor,
        cache_dir,
        **cfg,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        for key, value in cfg.items():
            setattr(self, key, value)
            
        self.processor = processor
        
        self.prediction = {}
        
        self.predict_language = True
        
        self.cfg_data_module = cfg_data_module
        
        self.vision_model = hydra.utils.instantiate(
            self.vision_model,
            cfg_data_module=cfg_data_module,
            processor=self.processor,
            cache_dir=cache_dir,
            _recursive_=False
        )
            
        self.language_model = hydra.utils.instantiate(
            self.language_model,
            cache_dir=cache_dir,
            _recursive_=False
        )

        self.all_predictions = {}
        self.all_losses = {}
        
        driving = None
        driving = DrivingAdaptor(
            self.language_model.hidden_size, 
            speed_wps_mode=self.speed_wps_mode,
            predict_route_as_wps=self.predict_route_as_wps,
        )

        self.adaptors = AdaptorList(
            language=LanguageAdaptor(self.language_model),
            driving=driving,
        )

        self.wp_encoder = WaypointInputAdaptor(
            token_size=self.language_model.hidden_size,
            hidden_size=256,
            hidden_size2=512,
            # norm_layer=NormZeroOne(min_max=(-32.0, 32.0)),
        )

        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor


    def forward(self,
        example: DrivingExample,
        return_language: Optional[bool] = None,
        prompt_ids: Optional[Tensor] = None,
    ) -> DrivingOutput:
        """
        Samples a trajectory from the model.
        """
        self.speed_wps, self.route, self.language = None, None, []
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
        
        if driving_input is not None:
            adaptor_dict = self.adaptors(example, inference=True)
            adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
                    adaptor_dict = adaptor_dict,
                    pixel_values = driving_input.camera_images,
                    placeholder_values = driving_input.prompt_inference.placeholder_values,
                    wp_encoder = self.wp_encoder,
                )
            
            input_embeds_all = adaptor_dict["language_inputs"]
            attention_masks = adaptor_dict['language_inputs_mask']


        if self.predict_language:
            # per batch item because of padding
            for b_idx, (input_embed, attention_mask) in enumerate(zip(input_embeds_all, attention_masks)):
                input_embed = input_embed.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                if self.language_model.variant == 'OpenGVLab/InternVL2-4B':
                    eos = self.tokenizer.added_tokens_encoder['<|end|>']
                elif self.language_model.variant == 'OpenGVLab/InternVL2-2B':
                    eos = self.tokenizer.added_tokens_encoder['<|im_end|>']
                else:
                    eos = self.tokenizer.eos_token_id

                # BUG: input_embeds, cot
                sampled_tokens, input_embeds = self.language_model.greedy_sample(
                    input_embed,
                    eos_token_id=eos,
                    max_new_tokens=100,
                    input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                    logit_matrix=self.adaptors.language.lm_head.weight,
                    attention_mask=attention_mask,
                    # position_ids=position_ids,
                )
                
                inputs_driving = self.adaptors.driving(driving_input)
                input_embed_concat = torch.cat((input_embeds, inputs_driving["inputs"][b_idx].unsqueeze(0)), dim=1)
                features, logits = self.language_model.forward(input_embed_concat)

                len_driving = inputs_driving["inputs"].size(1)

                driving_features = features[:, -len_driving:]
                driving_logits = logits[:, -len_driving:]
                predictions = self.adaptors.driving.get_predictions(driving_features, driving_logits)
                    
                for k, v in predictions.items():
                    if v is not None:
                        if hasattr(self, k) and getattr(self, k) is not None:
                            if isinstance(v, torch.Tensor):
                                setattr(self, k, torch.cat((getattr(self, k), v), dim=0))
                            elif isinstance(v, list):
                                getattr(self, k).append(v)
                            else:
                                raise NotImplementedError(f"Type of {k} not supported")
                        else:
                            setattr(self, k, v)
                                
                self.language.append(self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)[0])
        else:
            # single forward pass same as during training so we can use the same function
            features = self.forward_model(driving_input, adaptor_dict)
            outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
            predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor['driving'])

            for k, v in predictions.items():
                if v is not None:
                    setattr(self, k, v)

        return self.speed_wps, self.route, self.language


    def forward_model(self, 
                      driving_input: DrivingInput, 
                      adaptor_dict: Dict, 
                      driving_labels: DrivingLabel = None,
                    #   language_embeds: Tensor = None
                      ) -> Tensor:
        """
        Forward model conditioned on the given driving input.
        """
        
        adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict = adaptor_dict,
            pixel_values = driving_input.camera_images,
            placeholder_values = driving_input.prompt.placeholder_values,
            wp_encoder = self.wp_encoder,
        )

        position_ids = None
        adaptor_embeds = adaptor_dict["inputs"]
        adaptor_mask = adaptor_dict['inputs_mask']

        input_embeds = adaptor_embeds
        input_embeds = input_embeds.to(
            dtype=self.language_model.model.dtype
        )
        attention_mask = adaptor_mask

        outputs = self.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        features = outputs.hidden_states[-1]
        logits = outputs[0]

        vision_features, adaptor_features = features.split(
            [features.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        vision_logits, adaptor_logits = logits.split(
            [logits.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], dim=1
        )
        return adaptor_features, adaptor_logits
    

    def forward_loss(self, example: DrivingExample, per_sample=False) -> TrainingOutput:
        """
        Forward pass of the model for a driving input, followed by
        computing the next token cross-entropy loss.

        Args:
            driving_input: input to the vision encoder.
            text_ids: Text ids tensor of shape [B, T]. These are input to the model and used in the loss.
            text_mask: Text mask tensor of shape [B, T].
        """

        adaptor_dict = self.adaptors(example)
        adaptor_embeds = adaptor_dict["inputs"]
        adaptor_mask = adaptor_dict['inputs_mask']

        adaptor_features, adaptor_logits = self.forward_model(example.driving_input, adaptor_dict, driving_labels=example.driving_label)
        loss_dict = self.adaptors.compute_loss(adaptor_features, adaptor_logits, adaptor_dict, example)

        loss_dict_only_losses = {k:v for k, v in loss_dict.items() if k.endswith("loss")}
        loss_logs = {k:v for k, v in loss_dict.items() if k.endswith("log")}
        
        pred_labels = {k:v for k, v in loss_dict.items() if not k.endswith("loss") and not k.endswith("log")}
        if per_sample:
            return loss_dict_only_losses, pred_labels

        return summarise_losses(loss_dict_only_losses), loss_logs

    def training_step(self, batch: DrivingExample, _batch_idx: int = 0):
        output, loss_logs = self.forward_loss(batch)
        logs = output #.update(loss_logs)
        self.log_training_output(logs, "train")

        # log the loss
        self.log("train/loss", output.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}


    def validation_step(self, batch: DrivingExample, _batch_idx: int = 0):
        
        output, loss_logs = self.forward_loss(batch)
        logs = output #.update(loss_logs)
        self.log_training_output(logs, "val")

        # log the loss
        self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}

    def predict_step(self, batch: DrivingExample, _batch_idx: int = 0):
        run_ids = decode_uint8(batch.run_id)
        
        speed_wps, route, language = self.forward(batch, return_language=True)

        self.num_route_points = 20
        route_equal = []
        for i in range(len(route)):
            route_equal.append(self.equal_spacing_route(route[i].cpu()))
        route_equal = torch.tensor(route_equal)
        route = route_equal.to(route.device)
        
        
        route_gt = batch.driving_label.path
        speed_wps_gt = batch.driving_label.waypoints
        language_gt = batch.driving_label.answer.language_string
        
        if len(self.prediction) == 0:
            self.prediction = {
                "waypoints": [speed_wps],
                "route": [route],
                "language": language,
                "waypoints_gt": [speed_wps_gt],
                "route_gt": [route_gt],
                "language_gt": language_gt,
                "prompt": batch.driving_input.prompt.language_string,
                "path": run_ids,
                "qa_templates": batch.qa_templates,
                "eval_infos": batch.driving_label.eval_infos,
            }
        else:
            self.prediction["waypoints"].append(speed_wps)
            self.prediction["route"].append(route)
            self.prediction["language"].extend(language)
            self.prediction["waypoints_gt"].append(speed_wps_gt)
            self.prediction["route_gt"].append(route_gt)
            self.prediction["language_gt"].extend(language_gt)
            self.prediction["prompt"].extend(batch.driving_input.prompt.language_string)
            self.prediction["path"].extend(run_ids)
            self.prediction["qa_templates"].extend(batch.qa_templates)
            self.prediction["eval_infos"].extend(batch.driving_label.eval_infos)
            
        
        return speed_wps, route, language, speed_wps_gt, route_gt, language_gt

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

    def on_predict_epoch_end(self) -> None:    

        repo_path = get_original_cwd()

        if self.trainer.ckpt_path is not None:
            ckpt_path = Path(self.trainer.ckpt_path).parent.parent
        else:
            ckpt_path = Path(f'{repo_path}/outputs/{self.language_model.variant}')
        save_prediction_path = ckpt_path / "predictions"
        save_prediction_path.mkdir(exist_ok=True, parents=True)
        
        samples_cot = [i for i, l in enumerate(self.prediction["prompt"]) if "What should the ego do next?" in l]
        samples_qa = [i for i, l in enumerate(self.prediction["prompt"]) if "Q:" in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        language = [(l, l_gt, p) for l, l_gt, p in zip(self.prediction["language"], self.prediction["language_gt"], self.prediction["path"])]
        
        if len(samples_qa) > 0:
            # sort by templates
            sorted_samples = {} # question: {answer: [language, language_gt]}
            for qa_template, language_sample in zip(self.prediction["qa_templates"], language):
                question = qa_template[0]
                answer = qa_template[1]
                if question not in sorted_samples:
                    sorted_samples[question] = {}
                if answer not in sorted_samples[question]:
                    sorted_samples[question][answer] = []
                sorted_samples[question][answer].append(language_sample)
            
            if os.path.exists(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json"):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}_{time}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
            else:
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
        
        for samples, name in zip([samples_cot, samples_qa, samples_all], ["cot", "qa", "all"]):
            language_samples = [l for i, l in enumerate(language) if i in samples]
        
            # save language predictions
            save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(language_samples, f, indent=4)
            
        route_preds = self.prediction["route"]
        route_preds = torch.cat(route_preds, dim=0)
        route_gt = self.prediction["route_gt"]
        route_gt = torch.cat(route_gt, dim=0)
        
        waypoints_preds = self.prediction["waypoints"]
        waypoints_preds = torch.cat(waypoints_preds, dim=0)
        waypoints_gt = self.prediction["waypoints_gt"]
        waypoints_gt = torch.cat(waypoints_gt, dim=0)
        
        # calc distance between wps for 1d wps
        waypoints_preds_1d = []
        for i in range(len(waypoints_preds)):
            waypoint_pred = waypoints_preds[i]
            waypoints_preds_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_pred[i+1] - waypoint_pred[i]) for i in range(len(waypoint_pred)-1)])
            # cumsum to get the distance from the start
            waypoints_preds_1d_tmp = torch.cumsum(waypoints_preds_1d_tmp, dim=0)
            waypoints_preds_1d_tmp = [[x, 0] for x in waypoints_preds_1d_tmp]
            waypoints_preds_1d.append(waypoints_preds_1d_tmp)
        waypoints_preds_1d = torch.tensor(waypoints_preds_1d)
        
        waypoints_gt_1d = []
        for i in range(len(waypoints_gt)):
            waypoint_gt = waypoints_gt[i]
            waypoints_gt_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_gt[i+1] - waypoint_gt[i]) for i in range(len(waypoint_gt)-1)])
            # cumsum to get the distance from the start
            waypoints_gt_1d_tmp = torch.cumsum(waypoints_gt_1d_tmp, dim=0)
            waypoints_gt_1d_tmp = [[x, 0] for x in waypoints_gt_1d_tmp]
            waypoints_gt_1d.append(waypoints_gt_1d_tmp)
        waypoints_gt_1d = torch.tensor(waypoints_gt_1d)
        
        # calculate ade and fde for samples seperatly whihc have <SAFETY> in prompt and for <INSTRUCTION_FOLLOWING>
        samples_safety = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" in l]
        samples_instruction = [i for i, l in enumerate(self.prediction["prompt"]) if "<INSTRUCTION_FOLLOWING>" in l]
        samples_neither = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" not in l and "<INSTRUCTION_FOLLOWING>" not in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        
        ade_fde = {}
        
        def get_desired_end_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # one WP every 0.25 seconds
            # we want to get speed at the last WP
            last_wp = wps[-1] #.cpu().numpy()
            # we want the WP half second earlier than the last WP
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            prev_wp = wps[-1 - half_second] #.cpu().numpy()
            desired_speed = np.linalg.norm(prev_wp - last_wp) * 2.0
            return desired_speed
        def get_desired_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # one WP every 0.25 seconds
            # we want to get speed at the last WP
            # last_wp = wps[-1] #.cpu().numpy()
            # we want the WP half second earlier than the last WP
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            wp_half_second = wps[half_second] #.cpu().numpy()
            wp_one_second = wps[one_second] #.cpu().numpy()
            desired_speed = np.linalg.norm(wp_half_second - wp_one_second) * 2.0
            return desired_speed
        
        def get_desired_avg_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # one WP every 0.25 seconds
            # we want to get speed at the last WP
            # last_wp = wps[-1] #.cpu().numpy()
            # # we want the WP half second earlier than the last WP
            # one_second = int(carla_fps // (wp_freq))
            # half_second = one_second // 2
            first_wp = wps[0] #.cpu().numpy()
            last_wp = wps[-1] #.cpu().numpy()
            desired_speed = np.linalg.norm(first_wp - last_wp) / (len(wps) * 0.25)
            return desired_speed
        
        def get_1d_wps(wps):
            waypoints_1d = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(len(wps)-1)]
            # cumsum to get the distance from the start
            waypoints_1d = np.cumsum(waypoints_1d)
            waypoints_1d = [[x, 0] for x in waypoints_1d]
            
            # prepend 0,0
            waypoints_1d = [[0, 0]] + waypoints_1d
            
            return np.array(waypoints_1d).reshape(-1, 2)
        
            
        wp_freq = 5
        carla_fps = 20
            
        
        for samples, name in zip([samples_safety, samples_instruction, samples_neither, samples_all], ["instruction"]):
            if len(samples) == 0:
                continue
            route_preds_sample = route_preds[samples].cpu().numpy()
            route_gt_sample = route_gt[samples].cpu().numpy()
            waypoints_preds_sample = waypoints_preds[samples].cpu().numpy()
            waypoints_gt_sample = waypoints_gt[samples].cpu().numpy()
            eval_infos_sample = [self.prediction["eval_infos"][i] for i in samples]
            waypoints_org_sample = [eval_infos_sample[i]["org_wps"] for i in range(len(samples))]
            route_org_sample = [eval_infos_sample[i]["org_path"] for i in range(len(samples))]
            waypoints_instruction_sample = [np.array(eval_infos_sample[i]["new_wps"]) for i in range(len(samples))]
            route_instruction_sample = [eval_infos_sample[i]["new_path"] for i in range(len(samples))]
            prompts = [self.prediction["prompt"][i].replace("<IMG_CONTEXT>", "") for i in samples]
            pred_language = [self.prediction["language"][i] for i in samples]
            paths = [self.prediction["path"][i] for i in samples]
            
            success_rate_all = []
            success_rate_by_mode = {}
            success_rate_by_allowed = {}
            
            paths_by_mode = {}
            
            for i in range(len(samples)):
                mode = eval_infos_sample[i]["mode"]
                allowed = eval_infos_sample[i]['allowed']
                sample_path = paths[i]
                
                # Initialize mode in dictionary if not present
                if mode not in success_rate_by_mode:
                    success_rate_by_mode[mode] = []
                if mode not in paths_by_mode:
                    paths_by_mode[mode] = []
                
                # Initialize allowed in dictionary if not present
                if allowed not in success_rate_by_allowed:
                    success_rate_by_allowed[allowed] = []
                
                # get desired speed form WPs
                desired_end_speed_pred = get_desired_end_speed(waypoints_preds_sample[i])
                desired_end_speed_gt = get_desired_end_speed(waypoints_gt_sample[i])
                desired_end_speed_org = get_desired_end_speed(waypoints_org_sample[i])
                desired_end_speed_instruction = get_desired_end_speed(waypoints_instruction_sample[i])
                
                desired_speed_pred = get_desired_speed(waypoints_preds_sample[i])
                desired_speed_gt = get_desired_speed(waypoints_gt_sample[i])
                desired_speed_org = get_desired_speed(waypoints_org_sample[i])
                desired_speed_instruction = get_desired_speed(waypoints_instruction_sample[i])
                
                desired_avg_speed_pred = get_desired_avg_speed(waypoints_preds_sample[i])
                desired_avg_speed_gt = get_desired_avg_speed(waypoints_gt_sample[i])
                desired_avg_speed_org = get_desired_avg_speed(waypoints_org_sample[i])
                desired_avg_speed_instruction = get_desired_avg_speed(waypoints_instruction_sample[i])

                
                pred_wps_1d = get_1d_wps(waypoints_preds_sample[i])
                pred_wps_1d_diffs = np.diff(pred_wps_1d[:, 0])
                pred_speeds = pred_wps_1d_diffs / (wp_freq/carla_fps)
                
                org_wps_1d = get_1d_wps(waypoints_org_sample[i])
                org_wps_1d_diffs = np.diff(org_wps_1d[:, 0])
                org_speeds = org_wps_1d_diffs / (wp_freq/carla_fps)
                
                instruction_wps_1d = get_1d_wps(waypoints_instruction_sample[i])
                instruction_wps_1d_diffs = np.diff(instruction_wps_1d[:, 0])
                instruction_speeds = instruction_wps_1d_diffs / (wp_freq/carla_fps)
                
                x = np.arange(len(pred_speeds))*0.25
                
                # linear regression np
                slope_pred, intercept_pred = np.polyfit(x, pred_speeds, 1)
                slope_org, intercept_org = np.polyfit(x, org_speeds, 1)
                slope_instruction, intercept_instruction = np.polyfit(x, instruction_speeds, 1)
                
                current_speed = float(prompts[i].split("Current speed: ")[-1].split(" ")[0])
                
                if mode == 'stop':
                    # route doesnt matter
                    paths_by_mode[mode].append(sample_path)
                    if name == 'instruction' or name == 'neither':
                        if np.min(pred_speeds) < 0.1:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                            
                elif mode == 'slower':
                    paths_by_mode[mode].append(sample_path)

                    if name == 'instruction' or name == 'neither':
                        # forced instruction following
                        if slope_pred < (-0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                        
                elif mode == 'faster':
                    paths_by_mode[mode].append(sample_path)
                    
                    if name == 'instruction' or name == 'neither':
                        # forced instruction following
                        if slope_pred > (0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'target_speed':
                    paths_by_mode[mode].append(sample_path)
                    
                    try:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-2])
                    except:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-3])
                    # ade from pred to instruction WP should be closer than to the GT WP
                    # ade_pred_org = np.mean(np.linalg.norm(waypoints_preds_sample[i] - waypoints_org_sample[i], axis=-1))
                    # ade_pred_instruction = np.mean(np.linalg.norm(waypoints_preds_sample[i] - waypoints_instruction_sample[i], axis=-1))
                    if name == 'instruction' or name == 'neither':
                        if ((desired_end_speed_pred > 0.8 * desired_end_speed_instruction and desired_end_speed_pred < 1.2 * desired_end_speed_instruction) or (desired_end_speed_pred > 0.8 * target_speed and desired_end_speed_pred < 1.2 * target_speed)):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                    
                elif mode == 'lane_change':
                    paths_by_mode[mode].append(sample_path)
                    
                    # on path
                    fde_pred_org = np.linalg.norm(route_preds_sample[i][-1] - route_org_sample[i][-1], axis=-1)
                    fde_pred_instruction = np.linalg.norm(route_preds_sample[i][-1] - route_instruction_sample[i][-1], axis=-1)
                    if name == 'instruction' or name == 'neither':
                        if fde_pred_instruction < fde_pred_org:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'crash':
                    paths_by_mode[mode].append(sample_path)
                    
                    ade_path_org_instruction = np.mean(np.linalg.norm(route_org_sample[i] - route_instruction_sample[i], axis=-1))
                    ade_path_pred_org = np.mean(np.linalg.norm(route_preds_sample[i] - route_org_sample[i], axis=-1))
                    ade_path_pred_instruction = np.mean(np.linalg.norm(route_preds_sample[i] - route_instruction_sample[i], axis=-1))
                    if ade_path_org_instruction > 1.0:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < ade_path_pred_org:
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)
                    else:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < 1.0 and (np.mean(pred_speeds) < 1.3 * np.mean(instruction_speeds) or np.mean(pred_speeds) > 0.7 * np.mean(instruction_speeds)):
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)

                else:
                    print(f"Unknown mode: {mode} in sample {i} with path {sample_path}")
                                
            # save result per sample
            per_sample_results = {
                'paths_by_mode': paths_by_mode,
                'success_rate_by_mode': success_rate_by_mode
            }
            save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(per_sample_results, f, indent=4)
            
            if len(success_rate_all) > 0:
                total_success_rate = sum(success_rate_all) / len(success_rate_all)
                ade_fde.update({f"success_rate_total_{name}": total_success_rate})
            else:
                ade_fde.update({f"success_rate_total_{name}": 0})
                
            min_samples_per_mode = min([len(success_rate_by_mode[mode]) for mode in success_rate_by_mode])
            balanced_total_success_rate = 0
            # Calculate success rate for each mode
            for mode in success_rate_by_mode:
                if len(success_rate_by_mode[mode]) > 0:
                    success_rate = sum(success_rate_by_mode[mode]) / len(success_rate_by_mode[mode])
                    ade_fde.update({f"success_rate_{name}_{mode}": success_rate})
                else:
                    ade_fde.update({f"success_rate_{name}_{mode}": 0})
                
            ade_route = np.mean(np.linalg.norm(route_preds_sample - route_gt_sample, axis=-1), axis=-1)
            
            ade_fde.update({
                f"num_samples_{name}": len(ade_route),
            })

        save_path_tmp = f"{str(save_prediction_path)}/dreamer_results_rank_{self.local_rank}.json"
        if os.path.exists(save_path_tmp):
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path_tmp = f"{str(save_prediction_path)}/dreamer_results_rank_{self.local_rank}_{time}.json"
        with open(save_path_tmp, "w") as f:
            json.dump(ade_fde, f, indent=4)
        

    def log_training_output(self, training_output: TrainingOutput, mode: str, dataset: Optional[str] = None):
        losses = {k: n.detach() for k, n in training_output.loss_averages.items()}
        counts = {k: n.detach().sum() for k, n in training_output.loss_counts.items()}
        losses["loss"] = training_output.loss.detach()
        counts["loss"] = 1  # loss is already averaged
        for k, v in sorted(losses.items()):
            log_key = f"{mode}_losses/{k}"
            self.log(log_key, v, batch_size=counts[k], sync_dist=True, add_dataloader_idx=False)


    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=max_steps, pct_start=self.pct_start, verbose=False
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}
