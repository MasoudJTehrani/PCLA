from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype

from lead.common.constants import TransfuserBoundingBoxIndex
from lead.data_loader import carla_dataset_utils
from lead.inference import inference_utils
from lead.inference.config_open_loop import OpenLoopConfig
from lead.tfv6.center_net_decoder import PredictedBoundingBox
from lead.tfv6.planning_decoder import decode_two_hot
from lead.tfv6.tfv6 import Prediction, TFv6
from lead.training.config_training import TrainingConfig

np.set_printoptions(suppress=True)

LOG = logging.getLogger(__name__)


class OpenLoopInference:
    @beartype
    def __init__(
        self,
        config_training: TrainingConfig,
        config_open_loop: OpenLoopConfig,
        model_path: str,
        device: torch.device,
        prefix: str = "model",
    ):
        """
        Open-Loop-Inference constructor.

        Args:
            config_training: Training config object belong to model.
            config_open_loop: Open loop config object.
            model_path: Path to the trained model weights.
            device: Device to run inference on.
            prefix: Prefix of the model weights files to load.
        """
        self.config_training = config_training
        self.config_open_loop = config_open_loop
        self.device = device

        # Loading models
        self.nets: List[TFv6] = []
        for file in sorted(os.listdir(model_path)):
            if file.startswith(prefix) and file.endswith(".pth"):
                LOG.info(f"Loading model weight from {os.path.join(model_path, file)}")
                net = TFv6(self.device, self.config_training)
                if self.config_training.sync_batchnorm:
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                state_dict = torch.load(
                    os.path.join(model_path, file),
                    map_location=self.device,
                    weights_only=True,
                )

                if not config_open_loop.strict_weight_load:
                    # Drop any weights whose shapes don't match the current model (e.g., pos_emb length differences)
                    current_state = net.state_dict()
                    drop_keys = []
                    for k, v in state_dict.items():
                        if k in current_state and current_state[k].shape != v.shape:
                            drop_keys.append(k)
                    for k in drop_keys:
                        LOG.warning("Dropping mismatched weight %s: checkpoint %s vs model %s", k, state_dict[k].shape, current_state[k].shape)
                        state_dict.pop(k)
                    net.load_state_dict(state_dict, strict=False)
                else:
                    net.load_state_dict(state_dict, strict=True)
                net.cuda(device=self.device).eval()
                self.nets.append(net)
        self.step = 4  # Constant so produced images start with 5, not really important

    @beartype
    def ensemble_planning_decoder(
        self, predictions: List[Prediction]
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Ensemble the outputs of the planning decoder from multiple models.

        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_routes: The aggregated route.
            pred_future_waypoints: The aggregated future waypoints.
            pred_target_speed_scalar: The aggregated target speed.
            pred_target_speed_distribution: The aggregated target speed distribution.
        """
        pred_routes = pred_future_waypoints = pred_target_speed_scalar = pred_target_speed_distribution = (
            pred_future_headings
        ) = None

        if self.config_training.use_planning_decoder:
            if self.config_training.predict_target_speed:
                pred_target_speed_logits = torch.stack([pred.pred_target_speed_distribution[0] for pred in predictions]).mean(
                    dim=0, keepdim=True
                )  # Average target speed logits.

                pred_target_speed_distribution = F.softmax(pred_target_speed_logits, dim=-1)  # softmax probabilities.
                pred_target_speed_scalar = decode_two_hot(
                    pred_target_speed_distribution, self.config_training.target_speed_classes, self.device
                ).reshape(1, 1)  # Decode to scalar.
                if (
                    pred_target_speed_distribution[0, 0] > self.config_open_loop.brake_threshold
                ):  # Brake if we are confident enough.
                    pred_target_speed_scalar = torch.Tensor([0.0]).reshape(1, -1)
                if self.config_open_loop.lower_target_speed:  # Optionally lower the target speed.
                    pred_target_speed_scalar *= self.config_open_loop.lower_target_speed_factor

            if self.config_training.predict_temporal_spatial_waypoints:
                pred_future_waypoints = torch.stack([pred.pred_future_waypoints[0] for pred in predictions]).mean(
                    dim=0, keepdim=True
                )  # Average waypoints.

            if self.config_training.predict_spatial_path:
                pred_routes = torch.stack([pred.pred_route[0] for pred in predictions]).mean(
                    dim=0, keepdim=True
                )  # Average route.

            if self.config_training.use_navsim_data and predictions[0].pred_headings is not None:
                pred_future_headings = torch.stack([pred.pred_headings[0] for pred in predictions]).mean(
                    dim=0, keepdim=True
                )  # Average headings.

        return (
            pred_routes,
            pred_future_waypoints,
            pred_target_speed_scalar,
            pred_target_speed_distribution,
            pred_future_headings,
        )

    @beartype
    def ensemble_bounding_boxes(
        self, predictions: List[Prediction]
    ) -> Tuple[List[PredictedBoundingBox], List[PredictedBoundingBox]]:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            List of aggregated bounding boxes in vehicle system.
            List of aggregated bounding boxes in image system.
        """
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = [], []
        if self.config_training.detect_boxes:
            for prediction in predictions:
                pred_bb = prediction.pred_bounding_box.pred_bounding_box_vehicle_system.squeeze().reshape(-1, 9)
                if len(pred_bb) > 0:
                    pred_bounding_boxes_vehicle_system.append(pred_bb)

        if len(pred_bounding_boxes_vehicle_system) > 0:
            pred_bounding_boxes_vehicle_system = inference_utils.non_maximum_suppression(
                pred_bounding_boxes_vehicle_system, float(self.config_training.iou_treshold_nms)
            )

            pred_bounding_boxes_image_system = carla_dataset_utils.bb_vehicle_to_image_system(
                pred_bounding_boxes_vehicle_system,
                self.config_training.pixels_per_meter,
                self.config_training.min_x_meter,
                self.config_training.min_y_meter,
            )

            pred_bounding_boxes_vehicle_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_vehicle_system
            ]

            pred_bounding_boxes_image_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_image_system
            ]

        return pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system

    @beartype
    def ensemble_bev_semantic(
        self, predictions: List[Prediction]
    ) -> Optional[torch.Tensor]:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_bev_semantic: Tensor containing the aggregated BEV semantic map
        """
        if self.config_training.use_bev_semantic:
            pred_bev_semantic = []
            for prediction in predictions:
                pred_bev_semantic.append(prediction.pred_bev_semantic)
            stacked = torch.stack(pred_bev_semantic, dim=0)  # (num_models, num_batches, num_classes, H, W)
            ch0 = stacked[:, :, 0].min(dim=0).values.unsqueeze(1)  # (num_batches, 1, H, W)
            others = stacked[:, :, 1:].max(dim=0).values  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    @beartype
    def ensemble_depth(self, predictions: List[Prediction]) -> Optional[torch.Tensor]:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_depth: Tensor containing the aggregated depth map
        """
        if self.config_training.use_depth:
            pred_depth = []
            for prediction in predictions:
                pred_depth.append(prediction.pred_depth)
            stacked = torch.stack(pred_depth, dim=0)  # (num_models, num_batches, H, W)
            return stacked.mean(dim=0)  # (num_batches, H, W)
        return None

    @beartype
    def ensemble_semantic_segmentation(
        self, predictions: List[Prediction]
    ) -> Optional[torch.Tensor]:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_semantic: Tensor containing the aggregated semantic segmentation map
        """
        if self.config_training.use_semantic:
            pred_semantic = []
            for prediction in predictions:
                pred_semantic.append(prediction.pred_semantic)
            stacked = torch.stack(pred_semantic, dim=0)  # (num_models, num_batches, num_classes, H, W)
            ch0 = stacked[:, :, 0].min(dim=0).values.unsqueeze(1)  # (num_batches, 1, H, W)
            others = stacked[:, :, 1:].max(dim=0).values  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    @beartype
    def ensemble(self, _, predictions: List[Prediction]) -> OpenLoopPrediction:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        # Bounding boxes
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = (
            None,
            None,
        )
        if self.config_training.carla_leaderboard_mode:
            pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = self.ensemble_bounding_boxes(predictions)

        # BEV semantic map
        pred_bev_semantic = None
        if self.config_training.carla_leaderboard_mode:
            pred_bev_semantic = self.ensemble_bev_semantic(predictions)

        # Semantic segmentation
        pred_semantic = None
        if self.config_training.carla_leaderboard_mode:
            pred_semantic = self.ensemble_semantic_segmentation(predictions)

        # Depth
        pred_depth = None
        if self.config_training.carla_leaderboard_mode:
            pred_depth = self.ensemble_depth(predictions)

        # Planning
        pred_route, pred_future_waypoints, pred_target_speed_scalar, pred_target_speed_distribution, pred_future_headings = (
            self.ensemble_planning_decoder(predictions)
        )

        return OpenLoopPrediction(
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_future_headings=pred_future_headings,
            pred_route=pred_route,
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bev_semantic=pred_bev_semantic,
            pred_bounding_box_vehicle_system=pred_bounding_boxes_vehicle_system,
            pred_bounding_box_image_system=pred_bounding_boxes_image_system,
            pred_radar_predictions=None,
        )

    @beartype
    @torch.inference_mode()
    def forward(self, data: Dict[str, torch.Tensor]) -> OpenLoopPrediction:
        """Run inference on the ensemble of models.
        Args:
            data: Dictionary containing the input data for the model

        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        self.step += 1
        self.predictions: List[Prediction] = [net(data) for net in self.nets]
        return self.ensemble(data, self.predictions)

    def __getitem__(self, index):
        return self.nets[index]


@dataclass
class OpenLoopPrediction:
    """Raw output predictions from the open loop model."""

    pred_future_waypoints: Optional[np.ndarray]
    pred_future_headings: Optional[np.ndarray]
    pred_target_speed_scalar: Optional[np.ndarray]
    pred_target_speed_distribution: Optional[np.ndarray]
    pred_route: Optional[np.ndarray]
    pred_semantic: Optional[np.ndarray]
    pred_depth: Optional[np.ndarray]
    pred_bev_semantic: Optional[np.ndarray]
    pred_bounding_box_vehicle_system: List[PredictedBoundingBox] | None
    pred_bounding_box_image_system: List[PredictedBoundingBox] | None
    pred_radar_predictions: None
