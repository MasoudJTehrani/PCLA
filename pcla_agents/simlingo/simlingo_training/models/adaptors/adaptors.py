from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from simlingo_training.utils.custom_types import DrivingExample


def cross_track_error(points: Tensor, path: Tensor):
    """
    Computes the cross track error between a set of points and a path.

    Args:
        points: The set of points to compute the cross track error for with shape [b, n, 2].
        path: The path to compute the cross track error with with shape [b, m, 2]. The path
            can contain nan values which indicates that the path is not available for that position.

    Returns:
        The cross track error for each point in the set of points with shape [b, n].
    """

    points, path = points.float(), path.float()

    ind = torch.arange(path.size(0), device=path.device)[:, None]
    closest = torch.cdist(points, path).nan_to_num_(torch.inf).argmin(-1)
    pt0 = path[ind, (closest - 1).clamp_min(0)]
    pt1 = path[ind, closest]
    pt2 = path[ind, (closest + 1).clamp_max(path.size(1) - 1)]

    tangent = (pt2 - pt1).nan_to_num_(0.0) + (pt1 - pt0).nan_to_num_(0.0)
    normal = torch.stack((tangent[..., 1], -tangent[..., 0]), dim=-1)
    normal = normal / normal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-2)

    return (points - pt1).mul(normal).sum(-1).abs()

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
    
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WaypointInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, N, 2] and returns an output of shape [B, N, token_size]
    Args:
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
        norm_layer: the `Module` to use to normalize the values of the input tensor.
    """
    
    def __init__(
        self, token_size: int = 258, hidden_size: int = 64, hidden_size2: int = 128, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer

        self.mlp = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, hidden_size2), nn.ReLU(True), nn.Linear(hidden_size2, token_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input with dims [B, N, 2]

        Returns:
            Output with dims [B, N, token_size]
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x)
        return x


class DrivingAdaptor(nn.Module):
    def __init__(self, 
                hidden_size: int, 
                mlp_dim=256, 
                predict_route_as_wps=False, 
                speed_wps_mode=False,
            ):
        super().__init__()
        self.heads = {}
        self.order = []

        self.speed_wps_mode = speed_wps_mode
        self.predict_route_as_wps = predict_route_as_wps

        if predict_route_as_wps:
            self.future_waypoints = 20
            self.query_embeds_wps = nn.Parameter(0.02 * torch.randn((1, self.future_waypoints, hidden_size)))
            self.route_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim*2), nn.SiLU(True),nn.Linear(mlp_dim*2, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, 2, bias=False)
            )
            
            self.queries = {'route': self.query_embeds_wps}
            self.sizes = {'route': self.future_waypoints}
            self.heads["route"] = self.route_head
            self.order.append('route')

        if speed_wps_mode == '2d':
            dim = 2
        elif speed_wps_mode == '1d':
            dim = 1
        else:
            raise ValueError(f"speed_wps_mode must be '1d' or '2d', not {speed_wps_mode}")
        self.future_speed_waypoints = 10 #TODO: read from config
        self.query_embeds_speed = nn.Parameter(0.02 * torch.randn((1, self.future_speed_waypoints, hidden_size)))
        self.speed_wps_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, dim, bias=False)
            )
        self.heads["speed_wps"] = self.speed_wps_head
        self.queries['speed_wps'] = self.query_embeds_speed
        self.sizes['speed_wps'] = self.future_speed_waypoints
        self.order.append('speed_wps')


    def forward(self, 
            driving_example: DrivingExample,
            **kwargs
            ) -> Dict[str, Tensor]:

        try:
            driving_input = driving_example.driving_input
        except AttributeError:
            driving_input = driving_example
        
        b = driving_input.camera_images.shape[0]
        inputs = None

        for input_type in self.order:
            query_embed = self.queries[input_type]
            if inputs is None:
                inputs = query_embed.expand(b, -1, -1)
            else:
                inputs = torch.cat((inputs, query_embed.expand(b, -1, -1)), dim=1)

        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool)

        return {"inputs": inputs, "inputs_mask": inputs_mask}

    def get_predictions(
        self, 
        features: Tensor,
        logits: Optional[Tensor] = None
    ) -> Dict:

        current_index = 0
        predictions = {}
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]

            feature = features[:, current_index: current_index + size]
            prediction = self.heads[input_type](feature).cumsum(1)

            predictions[input_type] = prediction
            current_index += size
        
        return predictions


    def compute_loss(
        self, adaptor_features: Tensor, adaptor_logits: Tensor, _inputs: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        label = example.driving_label
        assert label is not None
        
        if self.predict_route_as_wps:
            label_route = label.path
        else:
            label_route = None

        if self.speed_wps_mode == '2d':
            label_speed_wps = label.waypoints[:, : self.future_waypoints + 1]
        elif self.speed_wps_mode == '1d':
            label_speed_wps = label.waypoints_1d
        else:
            label_speed_wps = None

        current_index = 0
        loss_dict = {}
        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]
            features_tmp = adaptor_features[:, current_index: current_index + size]
            label = locals()[f'label_{input_type}']

            prediction = self.heads[input_type](features_tmp).cumsum(1)
            loss = F.smooth_l1_loss(prediction, label, reduction="none").sum(-1)
            
            # if input_type == 'waypoints' and self.predict_route_as_wps:
            #     # compute cross track error
            #     cte = cross_track_error(prediction, label_waypoints)
            #     loss_dict[f"{input_type}_cte_loss"] = (cte, torch.ones_like(cte, dtype=torch.long))

            loss_dict[f"{input_type}_loss"] = (loss, torch.ones_like(loss, dtype=torch.long))
            loss_dict[f"{input_type}_prediction"] = prediction
            loss_dict[f"{input_type}_label"] = label
            current_index += size

        return loss_dict


class LanguageAdaptor(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.embed_tokens = language_model.model.embed_tokens
        if hasattr(language_model.model, "lm_head"):
            self.lm_head = language_model.model.lm_head
        elif hasattr(language_model.model, "embed_out"):
            self.lm_head = language_model.model.embed_out
        elif hasattr(language_model.model.base_model.model, 'output'):
            self.lm_head = language_model.model.base_model.model.output
        else:
            raise ValueError("Language model must have `lm_head` or `embed_out` attribute.")


    def forward(self, example: DrivingExample, inference=False, **kwargs) -> Dict[str, Tensor]:
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
            
        b = driving_input.camera_images.size(0)
        
        if inference:
            label = driving_input.prompt_inference
        else:
            label = driving_input.prompt
            
        if label is not None:
            ids = label.phrase_ids.long()
            ids_valid = label.phrase_valid  # true => is fed into model
            ids_mask = label.loss_masking # true => takes part in loss

        inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))
        return {"inputs": inputs, "inputs_mask": ids_valid, "_ids": ids, "_ids_mask": ids_mask}

    def compute_loss(
        self, adaptor_features: Tensor, adaptor_logits: Tensor, inputs: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        del example

        if adaptor_logits is None:
            adaptor_logits = self.lm_head(outputs[:, :-1])
        else:
            adaptor_logits = adaptor_logits[:, :-1]
        labels = torch.where(inputs["_ids_mask"], inputs["_ids"], -1)
        # Shift by 1 for next token prediction
        labels = labels[:, 1:]
        language_loss = F.cross_entropy(
            adaptor_logits.flatten(0, -2), labels.flatten(), ignore_index=-1, reduction="none"
        ).view_as(labels)
        return {"language_loss": (language_loss, labels.ne(-1))}

class AdaptorList(nn.Module):
    """
    Each adaptor is responsible for converting a driving example
    to a sequence of tokens and computing the loss on the token outputs.
    Adaptors are only used during training.
    """

    def __init__(
        self,
        driving: Optional[DrivingAdaptor] = None,
        language: Optional[LanguageAdaptor] = None,
    ):
        super().__init__()
        self.driving = driving
        self.language = language

    @property
    def adaptors(self):
        dct: Dict[str, Adaptor] = {}
        if self.language is not None:
            dct["language"] = self.language
        if self.driving is not None:
            dct["driving"] = self.driving
        return dct

    def forward(self, example: DrivingExample, **kwargs) -> Dict[str, Tensor]:
        """
        Construct input embeddings for the given driving example.
        """

        input_dict: Dict[str, Tensor] = {}
        inputs_list: List[Tensor] = []
        inputs_mask_list: List[Tensor] = []

        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = adaptor.forward(example, **kwargs)
            inputs_list.append(adaptor_input_dict["inputs"])
            inputs_mask_list.append(adaptor_input_dict["inputs_mask"])
            input_dict.update({key + "_" + k: v for k, v in adaptor_input_dict.items()})

        inputs = torch.cat(inputs_list, dim=1)
        inputs_mask = torch.cat(inputs_mask_list, dim=1)
        split_sizes = torch.as_tensor([x.size(1) for x in inputs_list])
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]

        # Apply random permutation of modalities during training
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1)
        # Apply permutation to move invalid tokens to end of sequence
        valid_perm = inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)
        perm = rand_perm.gather(1, valid_perm)

        input_dict["inputs"] = inputs[arange, perm]
        input_dict["inputs_mask"] = inputs_mask[arange, perm]
        input_dict["perm"] = perm
        input_dict["split_sizes"] = split_sizes
        return input_dict

    def compute_loss(
        self, features: Tensor, logits: Tensor, input_dict: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Distributes the output embeddings from the transformer to
        the correct loss function and returns a dictionary of losses.
        """

        features_by_adaptor = self.split_outputs_by_adaptor(input_dict, features)
        logits_by_adaptor = self.split_outputs_by_adaptor(input_dict, logits)

        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}

        # Compute loss in each adaptor
        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}
        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = _gather_from_dict(input_dict, key + "_")
            adaptor_features = features_by_adaptor[key]
            adaptor_logits = logits_by_adaptor[key]
            losses = adaptor.compute_loss(adaptor_features, adaptor_logits, adaptor_input_dict, example)
            loss_dict.update(losses)

        return loss_dict

    def split_outputs_by_adaptor(self, input_dict: Dict[str, Tensor], outputs: Tensor) -> Dict[str, Tensor]:
        """
        Splits the output tensor into the correct output for each adaptor, according to the
        split_sizes in the input_dict.
        """
        # First reverse permutation
        inv_perm = input_dict["perm"].argsort(-1)
        arange = torch.arange(inv_perm.size(0), device=inv_perm.device)[:, None]
        outputs = outputs[arange, inv_perm]

        # Now split output for each adaptor
        split_sizes = [int(x) for x in input_dict["split_sizes"]]
        outputs_list = list(outputs.split(split_sizes, dim=1))
        return {key: outputs_list[i] for i, key in enumerate(self.adaptors.keys())}


def _gather_from_dict(d: Dict[str, Tensor], prefix: str):
    out: Dict[str, Tensor] = {}  # dict comprehensions with if not supported
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out