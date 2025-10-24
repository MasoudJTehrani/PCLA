
import torch
from typing import Dict, Tuple, Optional
from torch import Tensor
from simlingo_training.utils.custom_types import TrainingOutput

def summarise_losses(
    loss_dict: Dict[str, Tuple[Tensor, Tensor]], weights: Optional[Dict[str, float]] = None
) -> TrainingOutput:
    """
    Computes the total loss from a dictionary of losses and their counts.

    The loss dict should contain two tensor for each key:
    - The loss value for each batch sample; shape [B].
    - The loss count for each batch sample; shape [B]. This is the number of items to average over, i.e.
      number of tokens, number of cuboids etc. For the case where each batch sample has a loss, you
      can set it to a ones tensor of shape [B].

    Optionally, a weights dictionary can be provided to weight the losses.

    Args:
        loss_dict: A dictionary of losses and their counts, for each batch sample.
        weights: A dictionary of weights for each loss key.

    Returns:
        A TrainingOutput object with the total loss and its components.
    """

    loss_values = {k: v for k, (v, _) in loss_dict.items()}
    loss_counts = {k: n for k, (_, n) in loss_dict.items()}
    loss_averages = {k: torch.where(n.sum() > 0, v.sum() / n.sum(), 0.0) for k, (v, n) in loss_dict.items()}
    if weights is None:
        loss = torch.stack(list(loss_averages.values())).sum()
    else:
        loss = torch.stack([weights.get(k, 1.0) * v for k, v in loss_averages.items()]).sum()
    return TrainingOutput(
        loss=loss,
        loss_values=loss_values,
        loss_counts=loss_counts,
        loss_averages=loss_averages,
    )