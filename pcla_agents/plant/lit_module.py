import logging

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy

from model import HFLM

logger = logging.getLogger(__name__)


class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.last_epoch = 0
        self.cfg_train = self.cfg.model.training
        self.model = HFLM(self.cfg.model.network, self.cfg)

        # Loss functions
        self.criterion_forecast = nn.CrossEntropyLoss()
        
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


        logits, targets, pred_wp, _ = self(batch)

        losses = {}

        # Waypoints
        losses["loss_wp"] = F.l1_loss(pred_wp, batch["waypoints"])

        # Forecast
        losses_forecast = [
            torch.mean(self.criterion_forecast(logits[i], targets[:, i]))
            for i in range(len(logits))
        ]
        losses["loss_forecast"] = torch.mean(torch.stack(losses_forecast))

        weights = {
            "loss_wp": 1,
            "loss_forecast": self.cfg.model.pre_training.get("forecastLoss_weight", 0),
        }

        loss_all = sum([loss*weights[name] for name, loss in losses.items()])

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
            ["x", "y", "yaw", "speed", "extent_x", "extent_y"]
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

            self.metrics_forecasting_acc[i](logits[i], targets[:, i])
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
        return


    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg_train.grad_norm_clip
        )
