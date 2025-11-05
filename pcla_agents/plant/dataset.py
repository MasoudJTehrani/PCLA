import os
import sys
import copy
import glob
import logging
import json
import gzip
import math
import numpy as np
from pathlib import Path
from beartype import beartype

import torch
from torch.utils.data import Dataset

from rdp import rdp
import carla_garage.transfuser_utils as t_u

class PlanTDataset(Dataset):
    @beartype
    def __init__(self, root: str, cfg, shared_dict=None, split: str = "all") -> None:
        self.cfg = cfg
        self.cfg_train = cfg.model.training
        self.data_cache = shared_dict

        # Class ids
        self.classes = {"car": 1.0,
                        "walker": 3.0,
                        "static": 4.0, # New classes
                        "static_car": 4.0
                        }
        
        self.traffic_signs = {"stop_sign" : 6.0, "traffic_light": 7.0}

        # Types for forecasting
        self.car_types = ["car", "walker"]

        self.labels = []
        self.measurements = []

        # If you're using a cluster check that recursive glob doesn't cause issues
        label_raw_path_all = glob.glob(root + "/**/boxes*", recursive=True)
        label_raw_path = [p[:-6] for p in label_raw_path_all]

        # label_raw_path = self.filter_data_by_town(label_raw_path_all, split)

        logging.info(f"Found {len(label_raw_path)} Route folders.")

        for route_dir in label_raw_path:
            route_dir = Path(route_dir)
            num_seq = len(os.listdir(route_dir / "boxes"))

            # ignore the first 5 and last two frames
            for seq in range(
                5,
                num_seq - self.cfg_train.pred_len - self.cfg_train.seq_len - 2,
            ):
                # load input seq and pred seq jointly
                label = []
                measurement = []
                for idx in range(
                    self.cfg_train.seq_len + self.cfg_train.pred_len
                ):
                    labels_file = route_dir / "boxes" / f"{seq + idx:04d}.json.gz"
                    measurements_file = (
                        route_dir / "measurements" / f"{seq + idx:04d}.json.gz"
                    )
                    label.append(labels_file)
                    measurement.append(measurements_file)

                self.labels.append(label)
                self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.labels       = np.array(self.labels      ).astype(np.bytes_)
        self.measurements = np.array(self.measurements).astype(np.bytes_)
        print(f"Loading {len(self.labels)} samples from {len(label_raw_path_all)} folders")


    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.measurements)


    def __getitem__(self, index):
        """Returns the item at index idx."""

        labels = self.labels[index]
        measurements = self.measurements[index]

        
        if not self.data_cache is None and labels[0] in self.data_cache:
            sample = self.data_cache[labels[0]]
        else:

            sample = {
                "input": [],
                "output": [],
            }

            loaded_labels = []
            loaded_measurements = []

            for i in range(self.cfg_train.seq_len + self.cfg_train.pred_len):
                measurements_i = json.load(gzip.open(measurements[i]))
                labels_i = json.load(gzip.open(labels[i]))

                loaded_labels.append(labels_i)
                loaded_measurements.append(measurements_i)

            # Ego waypoints
            matrices = np.array([x["ego_matrix"] for x in loaded_measurements[self.cfg_train.seq_len - 1 :]])
            reference = matrices[0]
            wps = np.linalg.inv(reference) @ matrices[1:]
            wps = wps[:, :2, 3].tolist()
            sample["waypoints"] = wps

            # Target point
            local_command_point = np.array(loaded_measurements[self.cfg_train.seq_len - 1]["target_point"])
            sample["target_point"] = tuple(local_command_point)

            # Forecasting offset
            offset = self.cfg.model.pre_training.future_timestep

            for sample_key, file in zip(
                ["input", "output"],
                [
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1],
                        loaded_labels[self.cfg_train.seq_len - 1],
                    ),
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1 + offset],
                        loaded_labels[self.cfg_train.seq_len - 1 + offset],
                    ),
                ],
            ):

                measurements_data = file[0]
                ego_matrix = np.array(measurements_data["ego_matrix"])
                ego_yaw = measurements_data['theta']
                if sample_key == "input":
                    ego_matrix_input = ego_matrix
                    ego_yaw_input = ego_yaw

                labels_data_all = file[1]

                labels_data = [x for x in labels_data_all if "class" in x and x["class"] in self.classes or x["class"] in self.traffic_signs]

                if sample_key == "input": # Filtering only on input, output gets filtered during matching
                    labels_data = [x for x in labels_data if x["distance"] < self.cfg_train.max_object_dist]

                # Transform pos and yaw to ego frame of input timestep (no change if sample_key=="input")
                pos = []
                yaw = []
                for labels_data_i in labels_data:
                    p = np.array(copy.deepcopy(labels_data_i["position"])) - np.array(labels_data_all[0]["position"])
                    p = np.append(p,[1])
                    p_global = ego_matrix @ p
                    p_t2 = np.linalg.inv(ego_matrix_input) @ p_global
                    pos.append(p_t2[:2])
                    yaw.append(labels_data_i["yaw"]+ego_yaw-ego_yaw_input)
                
                data_car = [ # This is the old variable name, it also contains pedestrians and static objects
                    [
                        self.classes[x["class"]],
                        float(pos[j][0]),
                        float(pos[j][1]),
                        float(t_u.normalize_angle_degree(np.rad2deg(yaw[j]))),
                        float(x["speed"] * 3.6) if "speed" in x else 0.0,
                        float(x["extent"][0]),
                        float(x["extent"][1]) if "OpensDoor" not in str(labels[0]) or x.get("role_name", "") != "scenario" else float(x["extent"][1])*2, # Increase y extent if opendoor scenario
                        float(x["id"]) if "id" in x else 1.23,
                    ]
                    for j, x in enumerate(labels_data)
                    if x["class"] in self.classes
                ]

                # Stop signs and traffic lights
                data_car += [
                    [
                        self.traffic_signs[x["class"]],
                        float(x["position"][0]),
                        float(x["position"][1]),
                        float(t_u.normalize_angle_degree(np.rad2deg(x["yaw"]))),
                        0,
                        float(x["extent"][0]),
                        float(x["extent"][1]),
                        1.23
                    ]
                    for x in labels_data if x["class"] in self.traffic_signs and x["affects_ego"] and ("state" not in x or x["state"] in ["Red", "Yellow"])
                ]

                # Forecasting
                if sample_key == "output":
                    # discretize box
                    if self.cfg.model.pre_training.quantize:
                        if len(data_car) > 0:
                            data_car = self.quantize_box(data_car)

                    # we can only use vehicles where we have the corresponding object in the input timestep
                    # if we don't have the object in the input timestep, we remove the vehicle
                    # if we don't have the object in the output timestep we add a dummy vehicle, that is not considered for the loss
                    data_car_by_id = {}
                    i = 0
                    for ii, x in enumerate(labels_data):
                        if x["class"] in self.car_types:
                            data_car_by_id[x["id"]] = data_car[i]
                            i += 1
                    data_car_matched = []
                    for i, x in enumerate(data_car_input):
                        input_id = x[7]
                        if input_id in data_car_by_id:
                            data_car_matched.append(data_car_by_id[input_id])
                        else:
                            # append dummy data
                            dummy_data = x
                            dummy_data[0] = 10.0  # type indicator for dummy
                            data_car_matched.append(dummy_data)

                    data_car = data_car_matched
                    assert len(data_car) == len(data_car_input)

                else:
                    data_car_input = data_car

                # remove id from data_car
                data_car = [x[:-1] for x in data_car]

                # Generate route boxes
                ego_extent_y = labels_data_all[0]["extent"][1]
                rdp_epsilon = 0.5

                waypoint_route = np.array(measurements_data["route_original"])
                shortened_route = rdp(waypoint_route, epsilon=rdp_epsilon)

                # convert points to vectors
                vectors = shortened_route[1:] - shortened_route[:-1]
                midpoints = shortened_route[:-1] + vectors/2.
                norms = np.linalg.norm(vectors, axis=1)
                angles = np.arctan2(vectors[:,1], vectors[:,0])

                data_route = []

                for i, midpoint in enumerate(midpoints):
                    data_route.append([2.0,
                                       float(midpoint[0]),
                                       float(midpoint[1]),
                                       float(t_u.normalize_angle_degree(np.rad2deg(angles[i]))),
                                       float(i),
                                       float(norms[i])/2,
                                       float(ego_extent_y),
                                       ])

                # we split route segments longer than 10m into multiple segments
                # improves generalization
                data_route_split = []
                for route in data_route:
                    if route[5] > 5:
                        routes = split_large_BB(
                            route, len(data_route_split)
                        )
                        data_route_split.extend(routes)
                    else:
                        data_route_split.append(route)
                data_route = data_route_split[: self.cfg_train.max_NextRouteBBs]

                if sample_key == "output":
                    data_route = data_route[: len(data_route_input)]
                    if len(data_route) < len(data_route_input):
                        diff = len(data_route_input) - len(data_route)
                        data_route.extend([data_route[-1]] * diff)
                    for x in data_route:
                        x[0] = 10 # Don't forecast route
                else:
                    data_route_input = data_route

                assert len(data_route) == len(
                    data_route_input
                ), "Route and route input not the same length"

                assert (
                    len(data_route) <= self.cfg_train.max_NextRouteBBs
                ), "Too many routes"

                if len(data_route) == 0:
                    # quit programm
                    print("ERROR: no route found")
                    logging.error("No route found in file: {}".format(file))
                    sys.exit()

                sample[sample_key] = data_car + data_route

            if not self.data_cache is None:
                self.data_cache[labels[0]] = sample

        assert len(sample["input"]) == len(
            sample["output"]
        ), "Input and output have different length"

        return sample

    
    def quantize_box(self, boxes):
        boxes = np.array(boxes)

        # range of xy is [-30, 30]
        # range of yaw is [-180, 180]
        # range of speed is [0, 120]
        # range of extent is [0, 30]

        # quantize xy
        boxes[:, 1] = (boxes[:, 1] + 30) / 60
        boxes[:, 2] = (boxes[:, 2] + 30) / 60

        # quantize yaw
        boxes[:, 3] = (boxes[:, 3] + 180) / 360

        # quantize speed
        boxes[:, 4] = boxes[:, 4] / 120

        # quantize extent
        boxes[:, 5] = boxes[:, 5] / 30
        boxes[:, 6] = boxes[:, 6] / 30

        boxes[:, 1:] = np.clip(boxes[:, 1:], 0, 1)

        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, [1, 2, 5, 6]] = (boxes[:, [1, 2, 5, 6]] * (size_pos - 1)).round()
        boxes[:, 3] = (boxes[:, 3] * (size_angle - 1)).round()
        boxes[:, 4] = (boxes[:, 4] * (size_speed - 1)).round()

        return boxes.astype(np.int32).tolist()

    # Used only for viz
    def unquantize_box(self, boxes):
        boxes = np.array(boxes).astype(np.float32)
        if len(boxes.shape) < 2:
            boxes = boxes[None, :]
        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, [1, 2, 5, 6]] = boxes[:, [1, 2, 5, 6]] / (size_pos - 1)
        boxes[:, 3] = boxes[:, 3] / (size_angle - 1)
        boxes[:, 4] = boxes[:, 4] / (size_speed - 1)

        # unquantize xy
        boxes[:, 1] = boxes[:, 1] * 60 - 30
        boxes[:, 2] = boxes[:, 2] * 60 - 30

        # unquantize yaw
        boxes[:, 3] = (boxes[:, 3] * 360) - 180

        # unquantize speed
        boxes[:, 4] = boxes[:, 4] * 120

        # unquantize extent
        boxes[:, 5] = boxes[:, 5] * 30
        boxes[:, 6] = boxes[:, 6] * 30

        return boxes.tolist()

    def filter_data_by_town(self, label_raw_path_all, split):
        # in case we want to train without T2 and T5
        label_raw_path = []
        if split == "train":
            for path in label_raw_path_all:
                if "Town02" in path or "Town05" in path:
                    continue
                label_raw_path.append(path)
        elif split == "val":
            for path in label_raw_path_all:
                if "Town02" in path or "Town05" in path:
                    label_raw_path.append(path)
        elif split == "all":
            label_raw_path = label_raw_path_all
            
        return label_raw_path

def split_large_BB(route, start_id):
    x = route[1]
    y = route[2]
    angle = -route[3] - 90 # TODO
    extent_x = route[5]
    extent_y = route[6]

    x1 = x - extent_x * math.sin(math.radians(angle))
    y1 = y - extent_x * math.cos(math.radians(angle))

    x0 = x + extent_x * math.sin(math.radians(angle))
    y0 = y + extent_x * math.cos(math.radians(angle))

    number_of_points = (
        math.ceil(extent_x * 2 / 10) - 1
    )  # 5 is the minimum distance between two points, we want to have math.ceil(extent_x / 5) and that minus 1 points
    xs = np.linspace(
        x0, x1, number_of_points + 2
    )  # +2 because we want to have the first and last point
    ys = np.linspace(y0, y1, number_of_points + 2)

    splitted_routes = []
    for i in range(len(xs) - 1):
        route_new = route.copy()
        route_new[1] = (xs[i] + xs[i + 1]) / 2
        route_new[2] = (ys[i] + ys[i + 1]) / 2
        route_new[4] = float(start_id + i)
        route_new[5] = route[5] / (
            number_of_points + 1
        )
        route_new[6] = extent_y
        splitted_routes.append(route_new)

    return splitted_routes

def generate_batch(data_batch):
    maxseq = max([len(sample["input"]) for sample in data_batch])
    B = len(data_batch)

    x_batch_objs = [[5, 0, 0, 0, 0, 0, 0]]  # Padding at idx 0
    y_batch_objs = [[10, 0, 0, 0, 0, 0, 0]]  # Padding

    x_batch_objs.append([0, 0, 0, 0, 0, 0, 0])  # WP (CLS) token at idx 1
    y_batch_objs.append([10, 0, 0, 0, 0, 0, 0])  # 10 for dummy, so it doesnt get forecasted
    maxseq += 1

    batch_idxs = torch.zeros((B, maxseq), dtype=torch.int32) # This reconstructs the batch from object vector

    keys = [x for x in data_batch[0] if x not in ["input", "output"]]

    batches = {key: [] for key in keys}

    n = 2  # Padding 0, WP 1

    for i, sample in enumerate(data_batch):
        # Input
        n_sample = len(sample["input"])

        batch_idxs[i, 0] = 1  # First is wp token
        batch_idxs[i, 1:n_sample+1] = torch.arange(n, n+n_sample)  # Rest is normal tokens

        n += n_sample

        x_batch_objs.extend(sample["input"])
        y_batch_objs.extend(sample["output"])

        for key in keys:
            batches[key].append(torch.tensor(sample[key], dtype=torch.float32))

    batches = {key: torch.stack(value) for key, value in batches.items()}
    batches["idxs"] = batch_idxs
    batches["x_objs"] = torch.tensor(x_batch_objs, dtype=torch.float32)
    batches["y_objs"] = torch.tensor(y_batch_objs, dtype=torch.long)

    # x_batch_objs[batch_idxs] needs to have an empty object in the first spot of every line (for wp forecasting)
    return batches

# Debugging
if __name__=="__main__":
    import yaml
    # Read YAML file
    with open("/home/simon/PlanTUpdate/config/config.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    with open("/home/simon/PlanTUpdate/config/model/PlanT.yaml", 'r') as stream:
        plnt = yaml.safe_load(stream)

    cfg["model"] = plnt
    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value

    cfg = DictAsMember(cfg)

    ds = PlanTDataset("/home/simon/PDM-Lite-DS/Town05", cfg)

    print(generate_batch([ds[15974]]))
