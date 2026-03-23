import logging

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy

from model import HFLM

from plant_variables import PlanTVariables

logger = logging.getLogger(__name__)

class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.plant_variables = PlanTVariables()

        self.wp_rep = self.cfg.model.waypoints.representation

        # self.last_epoch = 0
        self.cfg_train = self.cfg.model.training
        self.model = HFLM(self.cfg.model.network, self.cfg)

        # Loss functions
        self.criterion_speed = nn.CrossEntropyLoss() # weight & label smoothing
        self.criterion_forecast = nn.CrossEntropyLoss(ignore_index=-999)

        # Metrics
        self.metrics_forecasting_acc = nn.ModuleList(
            [Accuracy(task="multiclass", num_classes=classes) for classes in self.model.vocab_size]
        )

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.cfg.model.training)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[self.cfg.lrDecay_epoch, self.cfg.lrDecay_epoch + 10],
            gamma=0.1,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        waypoints_batch = batch["waypoints"]
        path_batch = batch["route"]
        path_batch = path_batch[..., :self.cfg.model.waypoints.path_len, :]

        targetspeed_batch = batch["target_speed"]

        logits, targets, pred_plan, _ = self(batch)

        losses = {}

        (pred_path, pred_wps, pred_speed) = pred_plan

        if pred_wps is not None:
            losses["loss_wp"] = F.l1_loss(pred_wps, waypoints_batch)

        if pred_path is not None:
            losses["loss_path"] = F.l1_loss(pred_path, path_batch)

        if pred_speed is not None:
            target_speeds = torch.tensor(self.plant_variables.target_speeds, device=targetspeed_batch.device)
            brake = torch.zeros_like(targetspeed_batch, dtype=torch.bool, device=targetspeed_batch.device)
            twohot_targs = self.get_two_hot_encoding(targetspeed_batch, target_speeds, brake)

            losses["loss_egospeed"] = self.criterion_speed(pred_speed, twohot_targs)

        losses_forecast = [
            torch.mean(self.criterion_forecast(logits[i], targets[i].squeeze()))
            for i in range(len(logits))
        ]
        losses["loss_forecast"] = torch.mean(torch.stack(losses_forecast))

        for name, loss in losses.items():
            self.log(
                f"train/{name}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

        weights = {
            "loss_wp": self.cfg.model.waypoints.get("wp_weight", 1), 
            "loss_forecast": self.cfg.model.pre_training.get("forecastLoss_weight", 0),
            "loss_path": self.cfg.model.waypoints.get("path_weight", 1),
            "loss_egospeed": self.cfg.model.waypoints.get("speed_weight", 1),
        }

        loss_all = sum([loss*weights[name] for name, loss in losses.items()])

        self.log(
            "train/loss_all",
            loss_all,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.cfg.gpus > 1,
            batch_size=self.cfg.model.training.batch_size,
        )

        for i, name in enumerate(
                ["x", "y", "yaw", "speed"]
            ):
                if i > self.model.num_attributes:
                    break
                self.log(
                    f"train/loss_{name}",
                    losses_forecast[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

                mask = targets[i].squeeze() != -999
                self.metrics_forecasting_acc[i](
                    logits[i][mask], targets[i][mask].squeeze()
                )
                self.log(
                    f"train/acc_{name}",
                    self.metrics_forecasting_acc[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

        return loss_all

    def validation_step(self, batch, batch_idx):
        pass

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg_train.grad_norm_clip
        )

    # # Torch version of get_two_hot_encoding in data.py which also works with batches
    # # target_speed Bx1, config_target_speeds C, brake Bx1
    def get_two_hot_encoding(self, target_speed, config_target_speeds, brake):
        if torch.any(target_speed < 0):
            raise ValueError('Target speed value must be non-negative for two-hot encoding.')
        
        # Calculate two-hot labes as described in https://arxiv.org/pdf/2403.03950.pdf
        labels = torch.zeros(target_speed.shape[0], len(config_target_speeds), device=target_speed.device)

        # Compare each target speed with the config speeds
        diffs = (config_target_speeds > target_speed[:, None]).float()
        vals, idxs = diffs.max(dim=1)

        # Doing this calculation for all rows and fixing the exceptions later
        upper_ind = idxs
        lower_ind = idxs - 1
        upper_val = config_target_speeds[upper_ind]
        lower_val = config_target_speeds[lower_ind]

        lower_weight = (upper_val-target_speed) / (upper_val - lower_val)
        upper_weight = (target_speed-lower_val) / (upper_val - lower_val)

        labels[torch.arange(target_speed.shape[0]), lower_ind] = lower_weight
        labels[torch.arange(target_speed.shape[0]), upper_ind] = upper_weight

        # Clear rows where brake or no config value greater than target speed
        labels[torch.logical_or(brake, vals==0)] = 0

        # Set brake rows to 0
        labels[brake, 0] = 1.0

        # Set rows with max speed and without brake pressed to last bin
        labels[torch.logical_and(vals==0, ~brake), -1] = 1.0

        return labels
