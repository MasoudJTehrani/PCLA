# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified from OmniDrive(https://github.com/NVlabs/OmniDrive)
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------

from sqlite3 import Timestamp
import torch
import torch.nn.functional as F
from mmcv.utils import auto_fp16
from mmcv.models import DETECTORS
import copy
import os
from mmcv.models.builder import build_head

from mmcv.core import bbox3d2result

from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.models.utils.grid_mask import GridMask

from mmcv.utils.misc import locations

from ...datasets.data_utils.constants import IGNORE_INDEX, EGO_WAYPOINT_TOKEN
from mmcv.models import builder

from ...utils.llava_llama import LlavaLlamaForCausalLM, add_special_token
from transformers import AutoTokenizer, GenerationConfig

from mmcv.utils.misc import load_model

from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import json
import mmcv
from mmcv.utils.misc import MLN
from mmcv.models.utils.transformer import inverse_sigmoid
from pathlib import Path
import time
import re
import numpy as np
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
from scipy.optimize import linear_sum_assignment
import cv2
from mmcv.utils import force_fp32, auto_fp16
from ..utils.freeze_module import freeze_module
from mmcv.models.utils import  DistributionModule, PredictModel,  \
                                CustomTransformerDecoder, CustomTransformerDecoderLayer, SinusoidalPosEmb, gen_sineembed_for_position, \
                                    linear_relu_ln, py_sigmoid_focal_loss
from mmcv.models.bricks import Linear
from mmcv.models.builder import HEADS 
import pickle
from diffusers.schedulers import DDIMScheduler
import matplotlib.pyplot as plt
from mmcv.utils.misc import memory_refresh
from mmcv.models.utils import build_transformer
from mmcv.models.builder import HEADS, build_loss 
from mmcv.models.utils.vis_utils import format_bbox, show_multicam_bboxes, draw_ld_vis
@DETECTORS.register_module()
class Orion(MVXTwoStageDetector):
    def __init__(self,
                 save_path='./results_vlm/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start = 1,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 map_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 lm_head=None,
                 tokenizer=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 use_lora=False,
                 pretrained=None,
                 fp16_infer=False, # for faster close-loop infer, infer without evaluation
                 fp16_eval=False,
                 fp32_infer=False,  # for infer without evaluation
                 fut_ts=6,
                 freeze_backbone=False,
                 use_col_loss = False,
                 use_gen_token=False,
                 use_critical_qa=False,
                 use_diff_decoder=False,
                 use_mlp_decoder=False,
                 plan_anchor_path=None,
                 diff_loss_weight=2.0,
                 ego_fut_mode=20,
                 with_bound_loss=True,
                 noise_x_offset=12,
                 noise_x_scale=24,
                 noise_y_offset=10,
                 noise_y_scale=40,
                 qa_pretrain=False,
                 temporal_prompt_input=False,
                 mix_qa_training=False,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
                 loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=1.0),
                 plan_cls_loss_smooth = False,
                 ):
        super(Orion, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.save_path = save_path
        self.mix_qa_training = mix_qa_training
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stride = stride
        self.use_col_loss = use_col_loss
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.query_pos = nn.Sequential(
            nn.Linear(396, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.fut_ts = fut_ts
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.ego_pose_pe = MLN(156)

        self.pts_bbox_head.query_pos = self.query_pos
        self.pts_bbox_head.time_embedding = self.time_embedding
        self.pts_bbox_head.ego_pose_pe = self.ego_pose_pe

        if map_head is not None:
            self.map_head = builder.build_head(map_head)
            self.map_head.query_pos = self.query_pos
            self.map_head.time_embedding = self.time_embedding
            self.map_head.ego_pose_pe = self.ego_pose_pe

        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=2048,
                                        padding_side="right",
                                        use_fast=False,
                                        )
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer = None
        
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if LID:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (self.position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.position_encoder = nn.Sequential(
                nn.Linear(depth_num*3, embed_dims*4),
                nn.ReLU(),
                nn.Linear(embed_dims*4, embed_dims),
            )
        
        use_critical_qa = use_critical_qa or qa_pretrain
        self.qa_pretrain = qa_pretrain
        if lm_head is not None:
            lm_kwargs = dict(use_gen_token=use_gen_token,use_critical_qa=use_critical_qa)
            self.lm_head = load_model(lm_head, use_lora, frozen, lm_kwargs, fp16_infer)
        if use_gen_token:
            add_special_token([EGO_WAYPOINT_TOKEN], tokenizer = self.tokenizer, model = self.lm_head)
            self.lm_head.config.waypoint_token_idx = self.tokenizer(EGO_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0]
        
        self.use_gen_token = use_gen_token
        self.use_diff_decoder = use_diff_decoder
        self.use_mlp_decoder = use_mlp_decoder
        assert self.use_gen_token if self.use_diff_decoder else True
        if self.use_gen_token:
            if not self.use_diff_decoder and not self.use_mlp_decoder: # use VAE to generate traj
                self.layer_dim = 4
                self.with_bound_loss = with_bound_loss
                self.with_cur = True
                # generator motion & planning
                self.present_distribution_in_channels = 4096
                self.future_distribution_in_channels = 4096+12
                self.now_pred_in_channels = 64
                self.PROBABILISTIC = True
                self.latent_dim = 32
                self.MIN_LOG_SIGMA = -5.0
                self.MAX_LOG_SIGMA = 5.0
                self.FUTURE_DIM = 6
                self.N_GRU_BLOCKS = 3
                self.N_RES_LAYERS = 3
                self.embed_dims = embed_dims
                self.ego_fut_mode = 6
                self.present_distribution = DistributionModule(
                    self.present_distribution_in_channels,
                    self.latent_dim,
                    min_log_sigma=self.MIN_LOG_SIGMA,
                    max_log_sigma=self.MAX_LOG_SIGMA,
                )

                self.future_distribution = DistributionModule(
                    self.future_distribution_in_channels,
                    self.latent_dim,
                    min_log_sigma=self.MIN_LOG_SIGMA,
                    max_log_sigma=self.MAX_LOG_SIGMA,
                )

                assert self.present_distribution_in_channels%self.layer_dim == 0
                self.predict_model = PredictModel(
                    in_channels=self.latent_dim,
                    out_channels=self.present_distribution_in_channels,
                    hidden_channels=int(self.present_distribution_in_channels/self.layer_dim),
                    num_layers=self.layer_dim
                )
                ego_fut_decoder = []
                for _ in range(2):
                    ego_fut_decoder.append(Linear(8192, 8192))
                    ego_fut_decoder.append(nn.ReLU())
                ego_fut_decoder.append(Linear(8192, self.ego_fut_mode*2))
                self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
                self.loss_plan_reg = build_loss(loss_plan_reg)
                self.loss_plan_bound = build_loss(loss_plan_bound)
                if self.use_col_loss:
                    self.loss_plan_col = build_loss(loss_plan_col)
                # self.loss_plan_dir = build_loss(loss_plan_dir)
                self.loss_vae_gen = build_loss(loss_vae_gen)
            
            elif self.use_diff_decoder:
                self.plan_cls_loss_smooth = plan_cls_loss_smooth
                self.diff_loss_weight = diff_loss_weight
                self.diff_traj_cls_loss_weight = 10.0
                self.diff_traj_reg_loss_weight = 8.0
                self.noise_x_offset = noise_x_offset
                self.noise_x_scale = noise_x_scale
                self.noise_y_offset = noise_y_offset
                self.noise_y_scale = noise_y_scale
                self.ego_fut_mode = ego_fut_mode
                with open(plan_anchor_path, 'rb') as f:
                    anchors = pickle.load(f)
                plan_anchor = np.array(anchors) # (20,6,2)
                assert self.ego_fut_mode == plan_anchor.shape[0]
                self.plan_anchor = nn.Parameter(
                    torch.tensor(plan_anchor, dtype=torch.float32),
                    requires_grad=False,
                ) # 20,6,2

                self.plan_anchor_encoder = nn.Sequential(
                    *linear_relu_ln(4096, 1, 1,512*6),
                    nn.Linear(4096, 4096),
                )
                self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(4096),
                    nn.Linear(4096, 4096),
                    nn.Mish(),
                    nn.Linear(4096, 4096),
                )
                diff_decoder_layer = CustomTransformerDecoderLayer(
                    num_poses=6,
                    d_model=4096,
                    d_ffn=4096,
                    num_head=32,
                )
                self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
                self.diffusion_scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="scaled_linear",
                    prediction_type="sample",
                )
            elif self.use_mlp_decoder: 
                self.waypoint_decoder = nn.Sequential(
                    nn.Linear(4096, 4096 // 2),
                    nn.GELU(),
                    nn.Linear(4096//2, 6*2),
                )
                self.waypoints_loss = nn.MSELoss(reduction='none')
        self.test_flag = False
        self.planning_metric = None
        if fp16_infer:
            self.img_backbone.half()
        self.fp16_infer = fp16_infer
        self.fp16_eval = fp16_eval
        assert fp16_infer if fp16_eval else True
        self.fp32_infer = fp32_infer
        assert not fp16_infer if fp32_infer else True

        self.freeze_backbone = freeze_backbone
        self.temporal_prompt_input = temporal_prompt_input

    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
        
    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None
        
    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B), C, H, W)


        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats

    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi

    def position_embeding(self, data, memory_centers, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['cam_intrinsic'].size(0)

        intrinsic = torch.stack([data['cam_intrinsic'][..., 0, 0], data['cam_intrinsic'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)

        return coords_position_embeding

    # @force_fp32(apply_to=('img'))
    def forward(self, data, return_loss=True):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            # return self.forward_train(**data)
            losses = self.forward_train(**data)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
            return outputs
        else:
            return self.forward_test(**data)
        
    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_attr_labels= None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      input_ids=None,
                      vlm_labels=None,
                      ego_fut_trajs = None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, # [(76,)]
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id) # (1, 76)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels, # [(76,)]
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX) # (1, 76)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length] # 2048
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length] # 2048
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id) # (1, 76)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None
        # img_metas = [img_metas[0][0]] # BUG:这样不是seq
        img_metas = [img_meta[0] for img_meta in img_metas]

        data['img_feats'] = self.extract_feat(data['img'])
        losses = self.forward_pts_train(gt_bboxes_3d, gt_labels_3d, gt_attr_labels,map_gt_bboxes_3d, map_gt_labels_3d, img_metas,input_ids, vlm_labels, vlm_attn_mask, ego_fut_trajs,**data)

        return losses


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_attr_labels,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,   
                          img_metas,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          ego_fut_trajs,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        B = data['img'].shape[0]
        location = self.prepare_location(img_metas, **data) # (6, 40, 40, 2)
        pos_embed = self.position_embeding(data, location, img_metas) # (1, 9600, 256)
        losses = dict()

        if self.with_pts_bbox:
            outs_bbox, det_query = self.pts_bbox_head(img_metas, pos_embed, **data) # (1, 257, 4096)
            vision_embeded_obj = det_query.clone()
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs_bbox, gt_attr_labels]
            if self.pts_bbox_head.pred_traffic_light_state:
                loss_inputs.append(data['traffic_state'])
                loss_inputs.append(data['traffic_state_mask'])
            if self.use_col_loss:
                loss, agent_outs = self.pts_bbox_head.loss(*loss_inputs)
            else:
                loss = self.pts_bbox_head.loss(*loss_inputs)
            losses.update(loss)
            
        if self.with_map_head:
            outs_lane, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            # reference vad trans
            device = gt_labels_3d[0].device
            map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_3d)
            lane_pts = [F.pad(map_gt_bboxes.fixed_num_sampled_points.to(device),(0,1)) for map_gt_bboxes in map_gt_vecs_list]
            loss_inputs = [lane_pts, map_gt_labels_3d, outs_lane, img_metas]

            if False:
                # for debug
                import pickle
                with open('lane_pts.pkl', 'wb') as file:
                    pickle.dump(lane_pts, file)
            losses.update(self.map_head.loss(*loss_inputs))

        if self.with_lm_head:
            if self.use_gen_token:
                vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
                vlm_loss, ego_feature = self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, use_cache=False, return_ego_feature=True)
                if self.mix_qa_training:
                    dummy_ego_feature = self.lm_head.get_model().embed_tokens(torch.tensor([[self.lm_head.config.waypoint_token_idx] for _ in range(B)]).cuda())
                    dummy_ego_feature = dummy_ego_feature.squeeze(1)
                    valid_input_mask = (input_ids == self.lm_head.config.waypoint_token_idx).sum(dim=-1).to(torch.bool)
                    dummy_ego_feature[valid_input_mask] = ego_feature
                    ego_feature = dummy_ego_feature
                    data['ego_fut_masks'][:,0,0] *= valid_input_mask.unsqueeze(-1)
                losses.update(vlm_loss=vlm_loss[0])
                current_states = ego_feature.unsqueeze(1)

                if not self.use_diff_decoder and not self.use_mlp_decoder:
                    distribution_comp = {}
                    noise = None
                    self.fut_ts = 6
                    if self.training:
                        future_distribution_inputs = ego_fut_trajs.reshape(B, ego_fut_trajs.shape[1], -1)
                    if self.PROBABILISTIC:
                        sample, output_distribution = self.distribution_forward(
                            current_states, future_distribution_inputs, noise
                        )
                        distribution_comp = {**distribution_comp, **output_distribution}

                    hidden_states = ego_feature.unsqueeze(1)
                    states_hs, future_states_hs = \
                        self.future_states_predict(B, sample, hidden_states, current_states)

                    ego_query_hs = \
                        states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                    ego_fut_trajs_list = []
                    for i in range(self.fut_ts):
                        outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
                        ego_fut_trajs_list.append(outputs_ego_trajs)

                    ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)
                    lane_scores = outs_lane['all_lane_cls_one2one'][-1]
                    lane_preds = outs_lane['all_lane_preds_one2one'][-1]
                    for p in range(self.map_head.n_control):
                        lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                        lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
                    lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
                    if self.with_bound_loss:
                        loss_plan_input = [ego_fut_preds, ego_fut_trajs[:,0], data['ego_fut_masks'][:,0,0], data['ego_fut_cmd'][:,0,0], lane_preds, lane_scores]
                    else:
                        loss_plan_input = [ego_fut_preds, ego_fut_trajs[:,0], data['ego_fut_masks'][:,0,0], data['ego_fut_cmd'][:,0,0]]
                    
                    if self.use_col_loss:
                        loss_planning_dict = self.loss_planning(*loss_plan_input, **agent_outs)
                    else:
                        loss_planning_dict = self.loss_planning(*loss_plan_input)
                    losses.update(loss_planning_dict)
                    loss_vae_gen = self.loss_vae_gen(distribution_comp, data['ego_fut_masks'][:,0,0])
                    loss_vae_gen = torch.nan_to_num(loss_vae_gen)
                    losses.update(loss_vae_gen=loss_vae_gen)
                elif self.use_diff_decoder:
                    bs = B
                    device = ego_feature.device
                    # 1. add truncated noise to the plan anchor
                    plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
                    odo_info_fut = self.norm_odo(plan_anchor)
                    timesteps = torch.randint(
                        0, 50,
                        (bs,), device=device
                    )
                    noise = torch.randn(odo_info_fut.shape, device=device)
                    noisy_traj_points = self.diffusion_scheduler.add_noise(
                        original_samples=odo_info_fut,
                        noise=noise,
                        timesteps=timesteps,
                    ).float()
                    noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
                    noisy_traj_points = self.denorm_odo(noisy_traj_points)

                    # debug visualization
                    # ============================================debug===========================================================
                    # self.noising_vis(self,plan_anchor,device)
                    # ============================================debug===========================================================
                    ego_fut_mode = noisy_traj_points.shape[1]
                    # 2. proj noisy_traj_points to the query
                    traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=512)
                   
                    traj_pos_embed = traj_pos_embed.flatten(-2)
                    traj_feature = self.plan_anchor_encoder(traj_pos_embed)
                    traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
                    # 3. embed the timesteps
                    time_embed = self.time_mlp(timesteps)
                    time_embed = time_embed.view(bs,1,-1)

                    # 4. begin the stacked decoder
                    poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, current_states, time_embed)
                    targets = torch.cumsum(ego_fut_trajs,dim=-2).squeeze(1)
                    trajectory_loss_dict = {}

                    lane_scores = outs_lane['all_lane_cls_one2one'][-1]
                    lane_preds = outs_lane['all_lane_preds_one2one'][-1]
                    for p in range(self.map_head.n_control):
                        lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                        lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
                    lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
                    for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
                        trajectory_cls_loss, trajectory_reg_loss, trajectory_bound_loss = self.loss_planning_diffusion(poses_reg, poses_cls, targets, plan_anchor, data['ego_fut_masks'][:,0,0],lane_preds, lane_scores)
                        trajectory_loss_dict[f"traj_diff_loss_cls_{idx}"] = trajectory_cls_loss
                        trajectory_loss_dict[f"traj_diff_loss_reg_{idx}"] = trajectory_reg_loss
                        trajectory_loss_dict[f"traj_diff_loss_bound_{idx}"] = trajectory_bound_loss
                        
                    losses.update(trajectory_loss_dict)
                elif self.use_mlp_decoder:
                    waypoint = self.waypoint_decoder(current_states)
                    waypoint = waypoint.reshape(-1,2)
                    wp_loss = self.waypoints_loss(waypoint.to(torch.float32), ego_fut_trajs.view(-1, 2).to(torch.float32))
                    if 'ego_fut_masks' in data: # ignore invalid fut trajs supervision
                        wp_loss = (wp_loss * data['ego_fut_masks'].view(-1, 1)).mean()
                    else:
                        wp_loss = wp_loss.mean()
                    wp_loss = torch.nan_to_num(wp_loss)
                    losses.update(wp_loss=wp_loss)
            else:
                waypoint = None
                vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
                vlm_loss= self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, use_cache=False)
                losses.update(vlm_loss=vlm_loss[0])
        return losses
    
    def forward_test(self, img_metas, **data):
        if not self.test_flag: #for interval evaluation
            if self.with_pts_bbox:
                self.pts_bbox_head.reset_memory()
            if self.with_map_head:
                self.map_head.reset_memory()
            self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'input_ids','gt_bboxes_3d','vlm_labels']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        B = 1
        mapped_class_names = [
        'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
        ]

        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        pos_embed = self.position_embeding(data, location, img_metas)
        bbox_results = []
        if self.with_pts_bbox:
            outs, det_query = self.pts_bbox_head(img_metas, pos_embed, **data)
            vision_embeded_obj = det_query.clone()
            if self.use_col_loss:
                bbox_list = self.pts_bbox_head.get_motion_bboxes(
                outs, img_metas)
                for bboxes, scores, labels, trajs in bbox_list:
                    bbox_result = bbox3d2result(bboxes, scores, labels)
                    bbox_result['trajs_3d'] = trajs.cpu()
                    bbox_results.append(bbox_result)
            else:
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_metas)
                for bboxes, scores, labels in bbox_list:
                    bbox_results.append(bbox3d2result(bboxes, scores, labels))
        
        lane_results = None 
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            lane_results = self.map_head.get_bboxes(outs, img_metas)
        generated_text = []
        metric_dict = {}
        if not (self.fp16_infer or self.fp32_infer) or self.fp16_eval :
            gt_attr_label = data['gt_attr_labels'][0].to('cpu')
            gt_bbox = data['gt_bboxes_3d'][0]
            fut_valid_flag = bool(data['fut_valid_flag'][0])
            gt_label = data['gt_labels_3d'][0].to('cpu')
            if self.use_col_loss:
                score_threshold = 0.6
                with torch.no_grad():
                    c_bbox_results = copy.deepcopy(bbox_results)
                    bbox_result = c_bbox_results[0]
                    # filter pred bbox by score_threshold
                    mask = bbox_result['scores_3d'] > score_threshold
                    bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
                    bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
                    bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
                    bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

                    # matched_bbox_result = self.assign_pred_to_gt_vip3d(
                    #     bbox_result, gt_bbox, gt_label)

                    # metric_dict = self.compute_motion_metric_vip3d(
                    #         gt_bbox, gt_label, gt_attr_label, bbox_result,
                    #         matched_bbox_result, mapped_class_names)
            
        if self.with_lm_head:
            history_input_output_id = []
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
            for i, input_ids in enumerate(data['input_ids'][0]):
                input_ids = input_ids.unsqueeze(0)
                special_token_inputs = False
                if not self.qa_pretrain:
                    if hasattr(self.lm_head.config,'waypoint_token_idx'):
                        if isinstance(self.lm_head.config.waypoint_token_idx,list):
                            for sptoken in self.lm_head.config.waypoint_token_idx:
                                if sptoken in input_ids:
                                    special_token_inputs = True
                                    break
                        else:
                            special_token_inputs = self.lm_head.config.waypoint_token_idx in input_ids
                if self.use_gen_token and special_token_inputs: # must be the final round conversation
                    history_input_output_id.append(input_ids)
                    context_input_ids = torch.cat(history_input_output_id,dim=-1)
                    ego_feature = self.lm_head.inference_ego(
                        inputs=context_input_ids,
                        images=vision_embeded,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True,
                        return_ego_feature=True
                    )
                    ego_feature = ego_feature.to(torch.float32)
                    current_states = ego_feature.unsqueeze(1)
                    if not self.use_diff_decoder and not self.use_mlp_decoder: # VAE-based generate 
                        distribution_comp = {}
                        noise = None
                        self.fut_ts = 6
                        if self.PROBABILISTIC:
                            # Do probabilistic computation
                            sample, output_distribution = self.distribution_forward(
                                current_states, None, noise
                            )
                            distribution_comp = {**distribution_comp, **output_distribution}

                        # 2. predict future state from distribution
                        hidden_states = ego_feature.unsqueeze(1)
                        states_hs, future_states_hs = \
                            self.future_states_predict(B, sample, hidden_states, current_states)

                        ego_query_hs = \
                            states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                        ego_fut_trajs_list = []
                        for i in range(self.fut_ts):
                            outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
                            ego_fut_trajs_list.append(outputs_ego_trajs)

                        ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)
                    elif self.use_diff_decoder:
                        step_num = 2
                        bs = ego_feature.shape[0]
                        device = ego_feature.device
                        self.diffusion_scheduler.set_timesteps(1000, device)
                        step_ratio = 20 / step_num
                        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
                        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

                        # 1. add truncated noise to the plan anchor
                        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
                        img = self.norm_odo(plan_anchor)
                        noise = torch.randn(img.shape, device=device)
                        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
                        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
                        noisy_trajs = self.denorm_odo(img)
                        ego_fut_mode = img.shape[1]
                        for k in roll_timesteps[:]:
                            x_boxes = torch.clamp(img, min=-1, max=1)
                            noisy_traj_points = self.denorm_odo(x_boxes)

                            # 2. proj noisy_traj_points to the query
                            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=512)
                            traj_pos_embed = traj_pos_embed.flatten(-2)
                            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
                            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

                            timesteps = k
                            if not torch.is_tensor(timesteps):
                                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
                            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                                timesteps = timesteps[None].to(img.device)
                            
                            # 3. embed the timesteps
                            timesteps = timesteps.expand(img.shape[0])
                            time_embed = self.time_mlp(timesteps)
                            time_embed = time_embed.view(bs,1,-1)

                            # 4. begin the stacked decoder
                            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, current_states, time_embed)
                            poses_reg = poses_reg_list[-1]
                            poses_cls = poses_cls_list[-1]
                            x_start = poses_reg[...,:2]
                            x_start = self.norm_odo(x_start)
                            img = self.diffusion_scheduler.step(
                                model_output=x_start,
                                timestep=k,
                                sample=img
                            ).prev_sample
                        mode_idx = poses_cls.argmax(dim=-1)
                        mode_masks = torch.zeros(*poses_cls.shape[:2],device=poses_cls.device)
                        for mask, idx in zip(mode_masks, mode_idx):
                            mask[idx] = 1
                        mode_masks = mode_masks.to(torch.bool)
                        # best_reg = poses_reg[mode_masks]
                        ego_fut_preds = poses_reg
                    elif self.use_mlp_decoder:
                        waypoint = self.waypoint_decoder(current_states)
                        waypoint = waypoint.reshape(-1,2)
                else:
                    history_input_output_id.append(input_ids)
                    context_input_ids = torch.cat(history_input_output_id,dim=-1)
                    output_ids = self.lm_head.generate(
                        inputs=context_input_ids,
                        images=vision_embeded,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True
                    )
                    generated_text.append(
                        dict(
                        Q=img_metas[0]['vlm_labels'].data[i],
                        A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True),
                    ))
                    history_input_output_id.append(output_ids)
 
            full_match = False
            if not self.qa_pretrain:
                if self.use_gen_token:
                    if not self.use_diff_decoder and not self.use_mlp_decoder:
                        mask_active_cmd = data['ego_fut_cmd'][:,0,0] == 1
                        ego_fut_preds_inactive = ego_fut_preds[~mask_active_cmd].to('cpu')
                        ego_fut_preds = ego_fut_preds[mask_active_cmd].flatten(0,1).to('cpu') # (6, 2)
                    elif self.use_diff_decoder:
                        mask_active_cmd = mode_masks
                        ego_fut_preds_inactive = ego_fut_preds[~mask_active_cmd].to('cpu')
                        ego_fut_preds = ego_fut_preds[mask_active_cmd].flatten(0,1).to('cpu') # (6, 2)
                    elif self.use_mlp_decoder:
                        ego_fut_preds = waypoint.to('cpu')
                        ego_fut_preds_inactive = None
                else:
                    traj = generated_text[0]['A'][0]
                    full_match = re.search(r'\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]', traj)

                    if full_match:
                        coordinates_matches = re.findall(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)', full_match.group(0))
                        coordinates = [tuple(map(float, re.findall(r'-?\d+\.\d+', coord))) for coord in coordinates_matches]
                        coordinates_array = np.array(coordinates)
                        ego_fut_preds = torch.tensor(coordinates_array)
                        if len(ego_fut_preds) != 6: # for unstable text outputs during close-loop evaluations
                            ego_fut_preds = torch.zeros(6,2)
                    else:
                        ego_fut_preds = torch.zeros(6,2) # for unstable text outputs during close-loop evaluations
                        full_match = True

            if self.use_gen_token or full_match:
                ego_fut_preds = ego_fut_preds.to(torch.float32) # for fp16 infer
                if not self.use_diff_decoder:
                    ego_fut_pred = ego_fut_preds.cumsum(dim=-2) 
                else:
                    ego_fut_pred = ego_fut_preds
                if not (self.fp16_infer or self.fp32_infer) or self.fp16_eval:
                    ego_fut_trajs = data['ego_fut_trajs'][0, 0]
                    ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)
                    metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                            pred_ego_fut_trajs = ego_fut_pred[None].to('cpu'),
                            gt_ego_fut_trajs = ego_fut_trajs[None].to('cpu'),
                            gt_agent_boxes = gt_bbox,
                            gt_agent_feats = gt_attr_label.unsqueeze(0),
                            fut_valid_flag = fut_valid_flag # 当前帧是否涵盖6个轨迹
                        )
                    metric_dict.update(metric_dict_planner_stp3)
                    lane_results[0]['fut_valid_flag'] = fut_valid_flag
                else:
                    metric_dict.update({'fut_valid_flag': False})
                    lane_results[0]['fut_valid_flag'] = False
                lane_results[0]['ego_fut_preds'] = torch.nan_to_num(ego_fut_pred)
                lane_results[0]['ego_fut_cmd'] = data['ego_fut_cmd']
                if os.getenv('DEBUG_SHOW_PRED', None) == "1":
                    show_dir = os.getenv('DEBUG_SHOW_PRED_DIR', None)
                    ego_fut_trajs_show = None
                    if 'ego_fut_trajs' in data:
                        ego_fut_trajs_show = data['ego_fut_trajs']
                    if not self.use_gen_token:
                        ego_fut_preds_inactive = None
                    img_to_show, img_bev, qa_img = self.show_results(data['img'].unsqueeze(0), img_metas, data, ego_fut_trajs=ego_fut_trajs_show,
                        waypoint=ego_fut_preds.unsqueeze(0).unsqueeze(0), waypoint_inactive=ego_fut_preds_inactive, bbox_result=bbox_list, 
                        lane_result=lane_results, metric_dict=metric_dict, use_gt=False, show_dir=show_dir, generated_text=None)
                    lane_results[0]['result_vis'] = dict(img_to_show=img_to_show, img_bev=img_bev, qa_img=qa_img)
            else:
                metric_dict.update({'fut_valid_flag': False})
                lane_results[0]['ego_fut_preds'] = torch.zeros((6, 2), dtype=torch.float32).to(location.device)
                lane_results[0]['ego_fut_cmd'] = data['ego_fut_cmd']
                lane_results[0]['fut_valid_flag'] = fut_valid_flag if not self.qa_pretrain else False

        return bbox_results, generated_text, lane_results, metric_dict
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_feat(data['img'])
        bbox_list = [dict() for i in range(len(img_metas))]
        if data['img'].dim() == 4: # (6,3,640,640)
            data['img'] = data['img'].unsqueeze(0)
        bbox_pts, generated_text, lane_results, metric_dict = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
            # print(result_dict['metric_results']['fut_valid_flag']) for debug
        bbox_list[0]['text_out'] = generated_text
        bbox_list[0]['pts_bbox'].update(lane_results[0])
       
        return bbox_list

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]

        odo_info_fut_x = 2*(odo_info_fut_x + self.noise_x_offset)/self.noise_x_scale -1 
        odo_info_fut_y = 2*(odo_info_fut_y + self.noise_y_offset)/self.noise_y_scale -1 
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * self.noise_x_scale - self.noise_x_offset
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * self.noise_y_scale - self.noise_y_offset
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag or pred_ego_fut_trajs.size(1)==6 :
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = np.nan_to_num(traj_L2)
                metric_dict['plan_obj_col_{}s'.format(i+1)] = np.nan_to_num(obj_coll.mean().item())
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = np.nan_to_num(obj_box_coll.mean().item())
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict
    
    def show_results(self, img, img_metas, data, map_gt_bboxes_3d=None,
                          map_gt_labels_3d=None, ego_fut_trajs=None, waypoint_inactive=None,
                          outs_bbox=None, outs_lane=None, bbox_result=None, lane_result=None,
                          waypoint=None, metric_dict=None, use_gt=False, show_dir=None, generated_text=None):
        tasks = [
            dict(
                task_name = 'class',
                num_out = 9,
                level = 'box',
                names =[
                    'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
                ]
            ),
        ]
        grid_config = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [1.0, 64.0, 1.0]}

        cams = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        cams_canvas_seq = [1, 0, 2, 4, 3, 5]
        intrinsics = data['cam_intrinsic'][0].cpu().numpy()
        lidar2img = data['lidar2img'][0].cpu().numpy()
        lidar2cam = np.linalg.inv(intrinsics) @ lidar2img
        cam2lidar = np.linalg.inv(lidar2cam)

        intrinsics = intrinsics[:, :3, :3]
        all_rots = cam2lidar[:, :3, :3]
        all_trans = cam2lidar[:, :3, 3]

        sample_idx = img_metas[0]['scene_token'] + '/frame_idx_' + str(img_metas[0]['frame_idx'])
        if not use_gt:
            if bbox_result is None:
                bbox_result = self.pts_bbox_head.get_motion_bboxes(outs_bbox, img_metas)
            if lane_result is None:
                lane_result = self.map_head.get_bboxes(outs_lane, img_metas)

        aug_imgs = img[0, 0].permute(0, 2, 3, 1).cpu().numpy() 
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        norm_mean = np.array(norm_mean).reshape((1, 1, 1, 3))
        norm_std = np.array(norm_std).reshape((1, 1, 1, 3))
        aug_imgs = aug_imgs[..., :3] * norm_std + norm_mean
        aug_imgs = aug_imgs * 255
        aug_imgs = aug_imgs[..., [2, 1, 0]] # cv2 write uses BGR image 
        aug_imgs = aug_imgs.astype(np.uint8)
        aug_imgs = np.ascontiguousarray(aug_imgs)

        all_imgs = []
        rots = []
        trans = []
        intrins = []
        for img_i in cams_canvas_seq:
            cam_name = cams[img_i]
            index = img_i
            img = aug_imgs[index]
            all_imgs.append(img)
            rots.append(all_rots[index])
            trans.append(all_trans[index])
            intrins.append(intrinsics[index])

        if not use_gt:
            bbox = format_bbox(bbox_result[0][0]).copy()
            bbox[:, [3, 4, 5]] = bbox[:, [4, 3, 5]]
            bbox[:, 6] = - (bbox[:, 6] + torch.pi / 2)
            bbox[:, 2] += bbox[:, 5] / 2 
            pred_dict = {
                'box' : bbox,
                'class' : bbox_result[0][2].detach().cpu().numpy(),
                'score' : bbox_result[0][1].detach().cpu().numpy(),
            }
        else:
            bbox = format_bbox(img_metas[0]['gt_bboxes_3d']).copy()
            bbox[:, [3, 4, 5]] = bbox[:, [4, 3, 5]]
            bbox[:, 6] = - (bbox[:, 6] + torch.pi / 2)
            bbox[:, 2] += bbox[:, 5] / 2 
            pred_dict = {
                'box' : bbox,
                'class' : format_bbox(img_metas[0]['gt_labels_3d'])
            }

        if show_dir is not None:
            os.makedirs(show_dir, exist_ok=True)

        img_to_show, img_bev, ratio = show_multicam_bboxes(all_imgs, intrins, rots, trans, cams,
                             grid_config, 1, tasks, pred_dict,
                             dataset_type='b2d')

        lidar2ego = img_metas[0]['lidar2ego']
        if not use_gt:
            lane_pts_3d = lane_result[0]['map_pts_3d'].detach().cpu().numpy()
            lane_pts_3d[..., 2] = -lidar2ego[2, 3]
            lane_labels_3d = lane_result[0]['map_labels_3d'].detach().cpu().numpy()
        else:
            lane_pts_3d = map_gt_bboxes_3d[0].fixed_num_sampled_points.detach().cpu().numpy()
            lane_pts_3d = np.concatenate(
                (lane_pts_3d, -np.ones_like(lane_pts_3d[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
            lane_labels_3d = map_gt_labels_3d[0].detach().cpu().numpy()
        ld_infos = {
            "lane_pts_3d": lane_pts_3d,
            "lane_labels_3d": lane_labels_3d,
            "class_names": ['Broken','Solid','SolidSolid','Center','TrafficLight','StopSign'],
            "grid_config": grid_config,
        }
        calib_infos = {
            "intrins": intrins,
            "rots": rots,
            "trans": trans,
        }
        img_to_show, img_bev = draw_ld_vis(
            all_imgs,
            calib_infos,
            cams,
            img_bev=img_bev,
            ratio=ratio,
            ld_infos=ld_infos,
            draw_aug_img=False,
            aug_imgs=aug_imgs,
        )
        
        commands = ["LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT", "CHANGE LANE RIGHT"]
        command = commands[int(data['command'].item())]
        img_bev = cv2.putText(img_bev, command, (20, 40), cv2.FONT_HERSHEY_TRIPLEX , 0.8, 1, 1, cv2.LINE_AA)

        if waypoint_inactive is not None:
            if len(waypoint_inactive.shape) < 4:
                waypoint_inactive = waypoint_inactive.unsqueeze(0)
            for i in range(waypoint_inactive.shape[1]):
                if not self.use_diff_decoder:
                    waypoint_pts = torch.nan_to_num(waypoint_inactive).cumsum(dim=-2) # sample 0, command i
                else:
                    waypoint_pts = torch.nan_to_num(waypoint_inactive)
                waypoint_pts = waypoint_pts[0, i:i+1].to(torch.float32).detach().cpu().numpy()
                waypoint_pts = np.concatenate(
                    (waypoint_pts, -np.ones_like(waypoint_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
                waypoint_infos = {
                    "lane_pts_3d": waypoint_pts,
                    "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
                    "class_names": ['waypointInactive'],
                    "grid_config": grid_config,
                }
                img_to_show, img_bev = draw_ld_vis(
                    all_imgs,
                    calib_infos,
                    cams,
                    img_bev=img_bev,
                    ratio=ratio,
                    ld_infos=waypoint_infos,
                    draw_aug_img=False,
                    aug_imgs=aug_imgs,
                )
            
        if ego_fut_trajs is not None:
            ego_fut_trajs_pts = ego_fut_trajs.cumsum(dim=-2)
            ego_fut_trajs_pts = ego_fut_trajs_pts[0].to(torch.float32).detach().cpu().numpy()
            ego_fut_trajs_pts = np.concatenate(
                (ego_fut_trajs_pts, -np.ones_like(ego_fut_trajs_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
            waypoint_infos = {
                "lane_pts_3d": ego_fut_trajs_pts,
                "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
                "class_names": ['waypointGT'],
                "grid_config": grid_config,
            }
            img_to_show, img_bev = draw_ld_vis(
                all_imgs,
                calib_infos,
                cams,
                img_bev=img_bev,
                ratio=ratio,
                ld_infos=waypoint_infos,
                draw_aug_img=False,
                aug_imgs=aug_imgs,
            )

        if waypoint is not None:
            if not self.use_diff_decoder:
                waypoint_pts = torch.nan_to_num(waypoint).cumsum(dim=-2)
            else:
                waypoint_pts = torch.nan_to_num(waypoint)
            waypoint_pts = waypoint_pts[0].to(torch.float32).detach().cpu().numpy()
            waypoint_pts = np.concatenate(
                (waypoint_pts, -np.ones_like(waypoint_pts[:, :, 0:1]) * lidar2ego[2, 3],), axis=-1,)
            waypoint_infos = {
                "lane_pts_3d": waypoint_pts,
                "lane_labels_3d": torch.zeros((1,)).to(torch.int).detach().cpu().numpy(),
                "class_names": ['waypoint'],
                "grid_config": grid_config,
            }
            img_to_show, img_bev = draw_ld_vis(
                all_imgs,
                calib_infos,
                cams,
                img_bev=img_bev,
                ratio=ratio,
                ld_infos=waypoint_infos,
                draw_aug_img=False,
                aug_imgs=aug_imgs,
            )
        
        img = np.concatenate([img_to_show, img_bev], axis=1)
        qa_img = np.ones([img.shape[0]//2, img.shape[1],3],dtype=np.uint8)*255
        if generated_text is not None and len(generated_text):
            scene_desc_list = []
            for qa in generated_text:
                question = 'Q: ' + qa['Q']
                answer = 'A: ' + qa['A'][0]
                for i in range(0,len(question),200):
                    scene_desc_list.append(question[i:i+200])
                for i in range(0,len(answer),200):
                    scene_desc_list.append(answer[i:i+200])
            y = 15
            for desc in scene_desc_list:
                qa_img = cv2.putText(qa_img, desc, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 1, cv2.LINE_AA)
                y += 20
            img = np.concatenate([img, qa_img],axis=0)
    
        if show_dir is not None:
            if use_gt:
                file_name = 'vis_' + sample_idx.replace('/', '_') + '_gt.jpg'
                output_path = os.path.join(show_dir, file_name)
            else:
                file_name = 'vis_' + sample_idx.replace('/', '_') + '.jpg'
                output_path = os.path.join(show_dir, file_name)
  
            text = file_name
            position = (10, 30)  
            font = cv2.FONT_HERSHEY_SIMPLEX  
            font_scale = 0.7  
            color = (255, 255, 255)  
            thickness = 2 
            cv2.putText(img, text, position, font, font_scale, color, thickness)
            cv2.imwrite(output_path, img)

        return img_to_show, img_bev, qa_img
    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result
    
    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,2]
        ignore_list = ['traffic_sign', 'traffic_cone',
                       'traffic_light', 'others', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                self.fut_mode = 6
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                best_mode_idxs = torch.argmax(pred_bbox['trajs_cls'][m_pred_idx] , dim=-1).tolist()
                pred_fut_trajs = pred_fut_trajs[best_mode_idxs]
                pred_fut_trajs = pred_fut_trajs[:num_valid_ts]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                # dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs[None, :, :], dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    def future_states_predict(self, batch_size, sample, hidden_states, current_states):

        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
        future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)

        hidden_states = hidden_states.permute(1,0,2) # (4, 1, 4096) -> (1, 4, 4096)
        hidden_state = hidden_states.reshape(self.layer_dim, -1, int(4096/4)) # (4, 4, 1024)
        future_states = self.predict_model(future_prediction_input, hidden_state)

        current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
        future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs

        return states_hs, future_states_hs

    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds = None,
                      lane_score_preds = None,
                      agent_preds= None,
                      agent_fut_preds= None,
                      agent_score_preds= None,
                      agent_fut_cls_preds= None
                      ):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        if lane_preds is not None and lane_score_preds is not None:
            loss_plan_bound = self.loss_plan_bound(
                ego_fut_preds[ego_fut_cmd==1],
                lane_preds,
                lane_score_preds,
                weight=ego_fut_masks,
                denormalize=False,
            )
        if self.use_col_loss:
            loss_plan_col = self.loss_plan_col(
                ego_fut_preds[ego_fut_cmd==1],
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
            )

        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = torch.nan_to_num(loss_plan_l1)
        if lane_preds is not None and lane_score_preds is not None:
            loss_plan_dict['loss_plan_bound'] = torch.nan_to_num(loss_plan_bound)
        if self.use_col_loss:
            loss_plan_dict['loss_plan_col'] = torch.nan_to_num(loss_plan_col)
            

        return loss_plan_dict
    

    def loss_planning_diffusion(self,
                      ego_fut_preds,
                      ego_fut_cls,
                      ego_fut_gt,
                      plan_anchor,
                      ego_fut_masks,
                      lane_preds = None,
                      lane_score_preds = None,
                    #   agent_preds,
                    #   agent_fut_preds,
                    #   agent_score_preds,
                    #   agent_fut_cls_preds
                      ):
        bs, num_mode, ts, d = ego_fut_preds.shape
        target_traj = ego_fut_gt
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - plan_anchor, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        mode_masks = torch.zeros(*ego_fut_cls.shape[:2],device=ego_fut_cls.device)
        for mask, idx in zip(mode_masks, mode_idx):
            mask[idx] = 1
        mode_masks = mode_masks.to(torch.bool)
        cls_target = mode_idx
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
        best_reg = torch.gather(ego_fut_preds, 1, mode_idx).squeeze(1)
        # import ipdb; ipdb.set_trace()
        # Calculate cls loss using focal loss
        target_classes_onehot = torch.zeros([bs, num_mode],
                                            dtype=ego_fut_cls.dtype,
                                            layout=ego_fut_cls.layout,
                                            device=ego_fut_cls.device)
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

        loss_plan_l1_weight = ego_fut_masks[:, :, None]
        
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1,  1, 2)

        loss_plan_l1 = self.diff_traj_reg_loss_weight * self.loss_plan_reg(
            best_reg,
            target_traj,
            loss_plan_l1_weight
        )

        if lane_preds is not None and lane_score_preds is not None:
            loss_plan_bound = self.loss_plan_bound(
                best_reg,
                lane_preds,
                lane_score_preds,
                weight=ego_fut_masks,
                denormalize=False,
            )

        
        if self.plan_cls_loss_smooth:
            loss_plan_cls_weight = torch.clip(dist, min=0, max=10.)*10 # scale factor
            loss_plan_cls_weight[mode_masks] = 10.
            loss_plan_cls_weight *= ego_fut_masks.all(dim=-1).to(torch.float).unsqueeze(-1)
            avg_factor = bs*self.ego_fut_mode
            loss_cls = self.diff_traj_cls_loss_weight * py_sigmoid_focal_loss(
                ego_fut_cls,
                target_classes_onehot,
                weight=loss_plan_cls_weight,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=avg_factor
            )
        else:
            loss_plan_cls_weight = ego_fut_masks.all(dim=-1).to(torch.float)
            loss_cls = self.diff_traj_cls_loss_weight * py_sigmoid_focal_loss(
                ego_fut_cls,
                target_classes_onehot,
                weight=loss_plan_cls_weight,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=None
            )

        return loss_cls, loss_plan_l1, loss_plan_bound
    
    def get_future_labels(self, gt_labels_3d, gt_attr_labels, ego_fut_trajs, device):

        agent_dim = 300
        veh_list = [0, 1, 2]
        mapped_class_names = [
            'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
        ]
        ignore_list = []
        batch_size = len(gt_labels_3d)
        gt_fut_trajs_bz_list = []

        for bz in range(batch_size):
            gt_fut_trajs_list = []
            gt_label = gt_labels_3d[bz]
            gt_attr_label = gt_attr_labels[bz]
            for i in range(gt_label.shape[0]):
                gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
                box_name = mapped_class_names[gt_label[i]]
                if box_name in ignore_list:
                    continue
                gt_fut_masks = gt_attr_label[i][self.fut_ts * 2:self.fut_ts * 3]
                num_valid_ts = sum(gt_fut_masks == 1)
                gt_fut_traj = gt_attr_label[i][:self.fut_ts * 2].reshape(-1, 2)
                gt_fut_traj = gt_fut_traj[:num_valid_ts]
                if gt_fut_traj.shape[0] == 0:
                    gt_fut_traj = torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)
                if gt_fut_traj.shape[0] < self.fut_ts:
                    gt_fut_traj = torch.cat(
                        (gt_fut_traj, torch.zeros([self.fut_ts - gt_fut_traj.shape[0], 2], device=device)), 0)
                gt_fut_trajs_list.append(gt_fut_traj)

            if len(gt_fut_trajs_list) != 0 & len(gt_fut_trajs_list) < agent_dim:
                gt_fut_trajs = torch.cat(
                    (torch.stack(gt_fut_trajs_list),
                     torch.zeros([agent_dim - len(gt_fut_trajs_list), self.fut_ts, 2], device=device)), 0)
            else:
                gt_fut_trajs = torch.zeros([agent_dim, self.fut_ts, 2], device=device)

            gt_fut_trajs_bz_list.append(gt_fut_trajs)

        if len(gt_fut_trajs_bz_list) != 0:
            gt_trajs = torch.cat((torch.stack(gt_fut_trajs_bz_list).repeat(1, 6, 1, 1), ego_fut_trajs), dim=1)
        else:
            gt_trajs = ego_fut_trajs

        return gt_trajs.reshape(batch_size, gt_trajs.shape[1], -1)

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):

        b = present_features.shape[0]
        c = present_features.shape[1]
        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.randn_like(present_mu)
        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise

        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution
