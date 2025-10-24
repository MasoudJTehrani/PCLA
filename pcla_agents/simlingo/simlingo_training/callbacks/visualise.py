import textwrap
from functools import wraps
from typing import Any, Callable, Dict

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from hydra.utils import get_original_cwd

from simlingo_training.utils.custom_types import DrivingExample

_STEPS_TO_FIRST_IDX: Dict[int, int] = {}


def get_1d_wps(wps):
    waypoints_1d = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(len(wps)-1)]
    # cumsum to get the distance from the start
    waypoints_1d = np.cumsum(waypoints_1d)
    waypoints_1d = [[x, 0] for x in waypoints_1d]
    
    # prepend 0,0
    waypoints_1d = [[0, 0]] + waypoints_1d
    
    return np.array(waypoints_1d).reshape(-1, 2)

def once_per_step(function: Callable[[Callback, Trainer, LightningModule, Any, Any, int], None]) -> Callable:
    """
    ``on_train_batch_end`` gets called by pytorch lightning an ``accumulate_grad_batches`` number of times per global step.
    Sometimes ``on_train_batch_end`` is intended per optimisation step, not per each forward pass of a batch.
    This wrapper provides such behaviour, in lack of having found an integrated pytorch lightning way so far*.

    Note:
        Wrapper specifically for `on_train_batch_end` from a pl `Callback`, in regards to the function signature.
    """

    # * `on_before_optimizer_step` is available but gets called before a step is finished, potentially leading to
    #   unexpected behaviour (e.g. report step timings that cut across steps, etc).
    # NOTE: The `_STEPS_TO_FIRST_IDX` global dict is not threadsafe, but `on_train_batch_end` is expected to only be
    #       called sequentially per process.
    # NOTE(technical): When `on_train_batch_end` is called, `trainer.global_step` is already updated. Hence taking the
    # first occurring `batch_idx` for a specific step is effectively the last `batch_idx` from the previous step and the
    # reporting thus is correct.
    @wraps(function)
    def only_on_first_go(
        self: Callback, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not all(
            (
                isinstance(self, Callback),
                isinstance(trainer, Trainer),
                isinstance(pl_module, LightningModule),
                isinstance(batch_idx, int),
            )
        ):
            raise ValueError(
                "Only use this decorator on `pl.Callback`'s `on_train_batch_end` function!",
            )
        global_step = trainer.global_step
        if global_step not in _STEPS_TO_FIRST_IDX:
            _STEPS_TO_FIRST_IDX[global_step] = batch_idx

        if _STEPS_TO_FIRST_IDX[global_step] == batch_idx:
            return function(self, trainer, pl_module, outputs, batch, batch_idx)
        return None

    return only_on_first_go


class VisualiseCallback(Callback):
    def __init__(self, interval: int, val_interval: int = 10, predict_interval: int = 1):
        super().__init__()
        self.interval = interval
        self.val_interval = val_interval
        self.predict_interval = predict_interval

    # @once_per_step
    @torch.no_grad
    def on_validation_batch_end(  # pylint: disable=too-many-statements
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        if trainer.global_step % self.val_interval != 0:
            return

        print("Validation visualization!")
        with torch.cuda.amp.autocast(enabled=True):
            # Forward with sampling
            # waypoints, route, target_speed, language = pl_module.forward(batch, return_language=True)
            speed_wps, route, language = pl_module.forward(batch, return_language=True)

        try:
            self._visualise_training_examples(batch, speed_wps, trainer, pl_module, 'val_waypoints')
            self._visualise_training_examples(batch, route, trainer, pl_module, 'val_route')
            # visualise_cameras(batch, pl_module, trainer, language, route, speed_wps, name='val_imgs')
            
            print("visualised_val_example")
            # _LOGGER.info("visualised_training_example")
        except Exception as e:  # pylint: disable=broad-except
            print("visualise_training_examples", e)
        #     pass
            # _LOGGER.exception("visualise_training_examples", e)
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            # Clear cache associated with decoding language model
            # _LOGGER.info("clearing_cache")
            pl_module.clear_cache()

    @once_per_step
    @torch.no_grad
    def on_train_batch_end(  # pylint: disable=too-many-statements
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        if trainer.global_step % self.interval != 0:
            return

        with torch.cuda.amp.autocast(enabled=True):
            # Forward with sampling
            speed_wps, route, language = pl_module.forward(batch, return_language=True)

        self._visualise_training_examples(batch, speed_wps, trainer, pl_module, 'waypoints', language_pred=language)
        self._visualise_training_examples(batch, route, trainer, pl_module, 'route', language_pred=language)
        # visualise_cameras(batch, pl_module, trainer, language, route, speed_wps, name='imgs')
    
        print("visualised_training_example")
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            pl_module.clear_cache()

    @rank_zero_only
    def _visualise_training_examples(
        self,
        batch: DrivingExample,
        waypoints,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        name: str,
        language_pred = None,
    ):
        if not pl_module.logger:
            return

        if 'waypoints' in name:
            waypoint_vis, prompt_img = visualise_waypoints(batch, waypoints, language_pred=language_pred)
        elif 'route' in name:
            waypoint_vis, prompt_img = visualise_waypoints(batch, waypoints, language_pred=language_pred, route=True)
        pl_module.logger.log_image(
            f"visualise/{name}", images=[Image.fromarray(waypoint_vis), prompt_img], step=trainer.global_step
        )
        plt.close("all")


def fig_to_np(fig):
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


@torch.no_grad()
def visualise_waypoints(batch: DrivingExample, waypoints, route=False, language_pred=None):
    assert batch.driving_label is not None
    repo_root = get_original_cwd()
    n = 11
    if route:
        n = 20
    fig = plt.figure(figsize=(10.24, 10.24))
    if route:
        gt_waypoints = batch.driving_label.path[:, :n, :].cpu().numpy()
    else:
        gt_waypoints = batch.driving_label.waypoints[:, :n].cpu().numpy()
    
    org_wps = []
    if len(batch.driving_input.prompt.placeholder_values) > 0:
        for b in batch.driving_input.prompt.placeholder_values:
            if 32007 in b:
                org_wps.append(np.array(b[32007]))
            elif 32005 in b:
                org_wps.append(np.array(b[32005]))

    pred_waypoints = waypoints[:, :n].cpu().numpy()
    b = gt_waypoints.shape[0]
    # visualise max 16 examples
    b = min(b, 16)
    rows = int(np.ceil(b / 4))
    cols = min(b, 4)

    white_pil = Image.new("RGB", (1024, 1024), "white")
    white_draw = ImageDraw.Draw(white_pil)

    # add space for text
    fig.subplots_adjust(hspace=0.8)
    y_curr = 10
    for i in range(b):
        if batch.driving_label.answer is not None:
            language = batch.driving_label.answer.language_string[i]

            wrapped_text = textwrap.fill(language, width=80) 
            wrapped_pred_text = textwrap.fill(language_pred[i], width=80) if language_pred is not None else ""

            lines_wrap = len(textwrap.wrap(wrapped_text, width=80))
            lines_wrap_pred = len(textwrap.wrap(wrapped_pred_text, width=80))
        
            white_draw.text((10, y_curr), f'{i} GT: {wrapped_text}', fill="black", font=ImageFont.truetype(f"{repo_root}/simlingo_training/arial.ttf", 20))
            y_curr += 20*lines_wrap
            white_draw.text((10, y_curr), f'{i} Pred: {wrapped_pred_text}', fill="black", font=ImageFont.truetype(f"{repo_root}/simlingo_training/arial.ttf", 20))
            y_curr += 20*lines_wrap_pred + 20
        ax = fig.add_subplot(rows, cols, i + 1)
        # Predicted waypoints
        ax.scatter(pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], marker="o", c="b")
        ax.plot(pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], c="b")
        # Ground truth waypoints (i.e. ideal waypoints)
        ax.scatter(gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], marker="x", c="g")
        ax.plot(gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], c="g")
        # Original waypoints
        if len(org_wps) > 0:
            ax.scatter(org_wps[i][:, 1], org_wps[i][:, 0], marker="o", c="r")
            ax.plot(org_wps[i][:, 1], org_wps[i][:, 0], c="r")
            
        ax.set_title(f"waypoints {i}")
        ax.grid()
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1.5)

    return fig_to_np(fig), white_pil
