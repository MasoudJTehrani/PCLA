# PlanTUpdate 🌱
In this folder you can find a simplified implementation of [PlanT](https://arxiv.org/abs/2210.14222), with upgrades for fair evaluation on the v2 version of Longest6. The modifications are described in E.1.1 in the [CaRL paper](https://arxiv.org/abs/2504.17838). You can find the original implementation [here](https://github.com/autonomousvision/plant/tree/main).

This folder provides the minimal code needed to train and evaluate the updated models and can be used stand-alone, making it perfect for use as a modern baseline or for further experimentation.

## Installation ⚙️
You can use the provided `environment.yml` to setup the conda environment. Afterwards we need to set some variables:
```shell
# Create conda env
conda env create -f environment.yml
conda activate PlanTUpdate

# TODO Replace your paths
LOCAL=/abs/path/to/CaRL/CARLA/original_leaderboard
CARLA=/abs/path/to/CARLA

conda env config vars set CARLA_ROOT=$CARLA -n PlanTUpdate
conda env config vars set LEADERBOARD_ROOT=$LOCAL/leaderboard -n PlanTUpdate
conda env config vars set SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner -n PlanTUpdate
conda env config vars set PYTHONPATH=$LOCAL/leaderboard:$LOCAL/scenario_runner:$CARLA/PythonAPI/carla -n PlanTUpdate

# Restart for variables to take effect
conda deactivate
conda activate PlanTUpdate
conda env config vars list
```

## Evaluation 🏎️
Lets get right into running PlanT on a route! You can download the trained checkpoints [here](https://1drv.ms/f/c/c3093771788ecd57/Eo5BiMk14qZCh2KG91g9pR4BsYBvORXvTBc_gQpT7YoeXA?e=7Degnz), then specify the paths in [eval.yaml](config/eval.yaml). Make sure to activate the conda environment before running the commands.
```shell
# Start CARLA in separate terminal, f.e. using:
$CARLA_ROOT/CarlaUE4.sh

# Run the evaluation using:
python ../CARLA/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes=data/longest6_split/longest6_00.xml --agent=PlanT_agent.py --agent-config=config/eval.yaml --track=MAP
```

To run a full evaluation of the Longest6 routes on a SLURM cluster, use [evaluate_routes_slurm.py](evaluate_routes_slurm.py):
```shell
python evaluate_routes_slurm.py --routes data/longest6_split --config config/eval.yaml --out_root results/longest6 --seeds 1 2 3 --retries 3
```

## Training 🏋️
You can train a model by running the `lit_train.py` script, you can find the relevant parameters in the [config.yaml](config/config.yaml) and [PlanT.yaml](config/model/PlanT.yaml) files. The dataset can be found on [HuggingFace](https://huggingface.co/datasets/autonomousvision/PDM_Lite_Carla_LB2/tree/main). Since it also contains RGB and LiDAR data, we provide a lighter, json-only version [here](https://1drv.ms/f/c/c3093771788ecd57/Eo5BiMk14qZCh2KG91g9pR4BsYBvORXvTBc_gQpT7YoeXA?e=7Degnz). If you prefer not to use WandB online, you can run `WANDB_MODE="offline" python lit_train.py`. If you're running the training on a cluster, you might want to change the path in `tmp_folder` to the node's SSD.

# Have fun experimenting! 🧪
If you have any questions or problems, feel free to open an issue :)


### Explanation of repo structure:
#### Folders
- `data` contains the route files for longest6
- `carla_garage` contains some useful functions from the [carla garage](https://github.com/autonomousvision/carla_garage/tree/leaderboard_2)
- `util` contains the old logging methods and the new visualization

#### Files
- `PlanT_agent.py` is the CARLA agent
- `data_testing.ipynb` can be used to visualize the training data and to confirm everything works
- `dataloader.py` contains the training dataloader and has minimal changes compared to the original repo
- `dataset.py` contains the dataset
- `lit_module.py` is the PyTorch Lightning module of the model
- `lit_train.py` is the training script
- `model.py` contains the PlanT model