
<div align="center">

# PCLA: Pretrained CARLA Leaderboard Agents

</div>

<p align="center">
<b>A framework for testing autonomous agents in the CARLA simulator</b> </br>
A versatile framework for deploying and evaluating autonomous driving agents (ADAs) from the CARLA Leaderboard on your own vehicle.
</p>

[![FSE 2025 Paper](https://img.shields.io/badge/Paper-FSE%202025-blue.svg)](https://dl.acm.org/doi/abs/10.1145/3696630.3728577)
[![CARLA Leaderboard](https://img.shields.io/badge/CARLA-Leaderboard-success.svg)](https://leaderboard.carla.org)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blueviolet.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/MasoudJTehrani/PCLA?style=social)](https://github.com/MasoudJTehrani/PCLA)
---

## Supported Autonomous Agents

PCLA currently supports **35** agents and 24 additional training seeds from 14 major autonomous driving projects:

[**SimLingo**](https://github.com/RenzKa/simlingo) | [**LMDrive**](https://github.com/opendilab/LMDrive) | [**TransfuserV3**](https://github.com/autonomousvision/transfuser) | [**TransfuserV4**](https://github.com/autonomousvision/carla_garage/tree/leaderboard_1) | [**TransfuserV5**](https://github.com/autonomousvision/carla_garage/tree/leaderboard_2) | [**TransfuserV6**](https://github.com/autonomousvision/lead) | [**CaRL**](https://github.com/autonomousvision/CaRL/tree/main/CARLA) | [**Roach**](https://github.com/autonomousvision/CaRL/tree/main/CARLA) | [**PlanT**](https://github.com/autonomousvision/CaRL/tree/main/PlanT) | [**Interfuser**](https://github.com/opendilab/InterFuser) | [**NEAT**](https://github.com/autonomousvision/neat) | [**WoR**](https://github.com/dotchen/WorldOnRails) | [**LBC**](https://github.com/dotchen/WorldOnRails) | [**LAV**](https://github.com/dotchen/LAV)

---

### Why Use PCLA?

PCLA (Pretrained CARLA Leaderboard Agents) is a versatile framework designed to decouple the autonomous driving agents (ADAs) from the restrictive CARLA Leaderboard codebase.

* **Decoupled Deployment:** Deploy high-performing ADAs onto a vehicle without relying on the original Leaderboard core codebase.
* **Easy Switching:** Effortlessly switch between **35 different agents** and their **24 additional training seeds** without requiring changes to CARLA versions or programming environments.
* **Version Independent:** Fully compatible with the latest version of CARLA, independent of the Leaderboard’s specific CARLA version.
* **Multi-Agent Support:** Run multiple vehicles with different autonomous agents simultaneously (note: requires high graphical memory).
* **CARLA Action Access:** Retrieve the computed CARLA movement action from the chosen agent for use in any custom application.

The corresponding paper is available at [Foundations of Software Engineering (FSE)](https://dl.acm.org/doi/abs/10.1145/3696630.3728577).

---

### Compatibility & Video

PCLA was tested on **Linux Ubuntu 22** and **CARLA 0.9.16 (Unreal Engine 4)**.

A video tutorial on how to use PCLA is available below (an updated version is coming soon).

[![PCLA Video Tutorial](https://img.youtube.com/vi/QyaMK6vclBg/0.jpg)](https://www.youtube.com/watch?v=QyaMK6vclBg)

---

## Table of Contents

1.  [Setup](#1-setup)
2.  [Pre-Trained Weights](#2-pre-trained-weights)
3.  [Autonomous Agents](#3-autonomous-agents)
4.  [How to Use](#4-how-to-use)
5.  [Navigation](#5-navigation)
6.  [Sample Code](#6-sample-code)
7.  [FAQ](#7-faq)
8.  [Citation](#8-citation)

---

## 1. Setup

### Prerequisites

1.  Download and install the [CARLA simulator](https://carla.readthedocs.io/en/latest/) from the official website (quick installation or build from source).
2.  Ensure **CUDA** and **PyTorch** are installed on your system.
    * [Tutorial for installing CUDA on Ubuntu](https://www.gpu-mart.com/blog/install-nvidia-cuda-11-on-ubuntu)
    * [Tutorial for PyTorch installation](https://pytorch.org/get-started/locally/)

### Installation Steps

1.  **Clone the repository and build the Conda environment:**

    ```bash
    git clone https://github.com/MasoudJTehrani/PCLA
    cd PCLA
    conda env create -f environment.yml
    conda activate PCLA
    ```
2.  **Install `torch-scatter` and CUDA Toolkit:**
    > **Note:** Please make sure to install `torch-scatter` and `CUDA Toolkit` according to your specific CUDA version. You can check your CUDA version using the included `python cuda.py` script.

3.  **Additional setup for LMDrive agent:**

    ```bash
    conda activate PCLA
    
    # 1. Vision Encoder
    cd pcla_agents/lmdrive/vision_encoder
    pip uninstall timm
    python setup.py develop
    
    # 2. LAVIS
    cd ../LAVIS
    python setup.py develop
    
    # 3. Fix ftfy dependency
    pip uninstall ftfy -y
    pip install "ftfy==6.1.1"
    cd ../../../
    ```

4.  **CARLA 0.9.16 Specific Installation:**
    If you intend to use PCLA with **CARLA 0.9.16**, you need to install the wheel from the `dist` folder:

    ```bash
    cd dist
    python3 -m pip install carla-0.9.16-cp38-cp38-linux_x86_64.whl
    ```
    > **Warning:** Some agents may act *weird* when used with CARLA 0.9.16.

---

## 2. Pre-Trained Weights

You have two options to download the required pre-trained model weights:

### Option 1: Automatic Download

Run the following script to automatically download and unzip the weights into the correct location:

```bash
python pcla_functions/download_weights.py
```
### Option 2: Manual Download

1.  Manually download the `pretrained.zip` file from [Hugging Face](https://huggingface.co/datasets/MasoudJTehrani/PCLA/blob/main/pretrained.zip).
    
2.  Extract the contents into the `PCLA/pcla_agents/` directory.
    

### Directory Structure

Ensure that the downloaded pre-trained weight folders are placed directly next to their respective model's folder. The final `pcla_agents` directory should look like this:

```
├── pcla_agents/
│   ├── transfuserpp/
│   ├── transfuserpp_pretrained/
│   ├── interfuser/
│   ├── interfuser_pretrained/
│   ├── ...
```
## 3. Autonomous Agents

PCLA includes **35** different autonomous agents and **24** additional training seeds to choose from. 
> **Find the repository of each agent at the top this page.**

### SimLingo

-   Also known as CarLLava.

-   `simlingo_simlingo`: The best-performing agent, which secured **first place** at [CARLA Leaderboard 2](https://leaderboard.carla.org) SENSORS track (previously named CarLLava).
        

### LMDrive

-   `lmdrive_llava`: Best performing LMDrive agent.
        
-   `lmdrive_vicuna`: Second best performing LMDrive agent.
        
-   `lmdrive_llama`: Third best performing LMDrive agent.
        

### TransfuserV3

-   Also known as Transfuser. See [the history of Transfuser](https://ln2697.github.io/lead/docs/transfuser_versions.html).

-   `tfv3_tf`: The main Transfuser agent.
        
-   `tfv3_ltf`: The LatentTF agent.
        
-   `tfv3_lf`: The Late_Fusion agent.
        
-   `tfv3_gf`: The Geometric_Fusion agent.
        

### TransfuserV4

-   Also know as Transfuser++ for Leaderboard 1.

-   **Seeds:** Replace `#` with the seed number from **0 to 2** (e.g., `tfpp_l6_0`).
        
-   `tfv4_l6_#`: Best performing Transfuser++ agent. Second place at CARLA Leaderboard 2 SENSORS track.
        
-   `tfv4_lav_#`: Transfuser++ not trained on Town02 and Town05.
        
-   `tfv4_wp_#`: Transfuser++ WP from their paper's appendix.
        
-   `tfv4_aim_#`: Reproduction of the [AIM](https://openaccess.thecvf.com/content/CVPR2021/html/Prakash_Multi-Modal_Fusion_Transformer_for_End-to-End_Autonomous_Driving_CVPR_2021_paper.html) method.
        
### TransfuserV5

-   Also known as Transfuser++ for Leaderboard 2. This version has a bit of similar performance as TransfuserV4

-   `tfv5_alltowns`: This agent is trained with all towns.

-   `tfv5_notown13`: This agent is trained exculding Town13.

### TransfuserV6

-   The most recent Transfuser agent, also known as Lead.

-   `tfv6_regnet`: Their best-performing agent that uses regnety032.

-   `tfv6_resnet`: Second best-performing agent that uses resnet34.

-   `tfv6_4cameras`: Uses 4cameras and resnet34.

-   `tfv6_noradar`: Uses resnet34 but no radar sensor.

-   `tfv6_visiononly`: Vision-only driving model and resnet34.

-   `tfv6_notown13`: Uses resnet34 but Town13 is excluded.

### CaRL

-   `carl_carl_#`: CaRL agent with a driving score of 64. Replace `#` with **0 or 1**.
        
-   `carl_carlv11`: The best CaRL agent with a driving score of 73. Best open-source RL planner on longest6 v2 and nuPlan.


### Roach

-   `carl_roach_#`: The Roach planner agent ([paper](https://arxiv.org/abs/2108.08265)) reproduced by the authors of [CaRL](https://github.com/autonomousvision/CaRL/tree/main). Replace `#` with a number from **0 to 4** for the 5 available seeds.


### PlanT

-   `carl_plant_#`: The PlanT planner agent ([paper](https://arxiv.org/abs/2210.14222)) reproduced by the authors of [CaRL](https://github.com/autonomousvision/CaRL/tree/main). Replace `#` with a number from **0 to 4** for the 5 available seeds.


#### NEAT
  
-   `neat_neat`
        
-   `neat_aimbev`
        
-   `neat_aim2dsem`
        
-   `neat_aim2ddepth`


####  Interfuser
 
-   `if_if`: Second best performing [CARLA Leaderboard 1](https://leaderboard.carla.org) SENSORS track agent.

#### Learning from All Vehicles (LAV)

-   `lav_lav`: The original LAV agent.
        
-   `lav_fast`: Leaderboard submission optimized for inference speed with temporal LiDAR scans.
        

#### Learning By Cheating (LBC)

-   `lbc_nc`: Learning By Cheating, the NoCrash model.
        
-   `lbc_lb`: Learning By Cheating, the Leaderboard model.
        

#### World on Rails (WoR)

-   `wor_nc`: World on Rails, the NoCrash model.
        
-   `wor_lb`: World on Rails, the Leaderboard model.


----------

## 4. How to Use

### Step 1: Run CARLA

Start the CARLA simulator. You **only** need the `-vulkan` flag for LBC, WoR, and LAV agents.

```Bash
./CarlaUE4.sh -vulkan
```

### Step 2: Integrate PCLA in Your Script

Open a new terminal and run your custom Python code. To use PCLA, import the library, define your agent and route, and initialize the `PCLA` class.

**Core Usage Snippet:**

```Python
from PCLA import PCLA

# 1. Define Agent and Route
agent = "tf_tf" # e.g., Transfuser
route = "./sample_route.xml"

# 2. Initialize PCLA (requires a CARLA client and a vehicle object)
pcla = PCLA(agent, vehicle, route, client)

# 3. Get action and apply control (within your simulation loop)
ego_action = pcla.get_action()
vehicle.apply_control(ego_action)

# 4. Cleanup when done
pcla.cleanup()

```

### Explaining the Arguments:

-   **`agent`**: Your chosen autonomous agent string (e.g., `"tf_tf"`). See [Autonomous Agents](https://www.google.com/search?q=%233-autonomous-agents).
    
-   **`route`**: The path to an XML file defining the vehicle's route, formatted according to the Leaderboard waypoints.
    
-   **`client` & `vehicle`**: The standard CARLA client and the ego-vehicle actor you wish to control.
    

----------

## 5. Navigation

PCLA provides utility functions to help you generate waypoints and routes usable by the agents.

### 5.1 Finding Spawn Points

To see all available spawn points and their associated index numbers in your current map:

Shell

```
python pcla_functions/spawn_points.py
```

### 5.2 Generating Waypoints

Use the `location_to_waypoint()` method to generate a sequence of CARLA waypoints between two CARLA locations.

```Python
from PCLA import location_to_waypoint
import carla

world = client.get_world()
vehicle_spawn_points = world.get_map().get_spawn_points() # Get CARLA spawn points
startLoc = vehicle_spawn_points[31].location              # Define Start location
endLoc = vehicle_spawn_points[42].location                # Define End location

# Returns a list of CARLA waypoints between the two locations
waypoints = location_to_waypoint(client, startLoc, endLoc) 

```

### 5.3 Creating the PCLA Route XML

Pass the generated CARLA waypoints to `route_maker()` to format them into a Leaderboard-compliant XML file that PCLA can read.

```Python
from PCLA import route_maker

route_maker(waypoints, "route.xml")
```

### Full Navigation Example

This combined example demonstrates generating an XML route from two [CARLA locations](https://carla.readthedocs.io/en/latest/python_api/#carlalocation):

```Python
from PCLA import route_maker, location_to_waypoint
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

vehicle_spawn_points = world.get_map().get_spawn_points()
startLoc = vehicle_spawn_points[31].location
endLoc = vehicle_spawn_points[42].location

# 1. Generate waypoints
waypoints = location_to_waypoint(client, startLoc, endLoc)

# 2. Create the PCLA XML route file
route_maker(waypoints, "route.xml")

```

----------

## 6. Sample Code

A comprehensive sample script is provided to help you test PCLA immediately.

To run the sample (which uses the LAV agent in Town02):

```Bash
python sample.py
```

> **Attention:** You may need to change the vehicle spawn point's index number on **line 45** of `sample.py` based on your specific CARLA version.

----------

## 7. FAQ

Frequently asked questions and possible issues are addressed and solved in [the PCLA issues section](https://github.com/MasoudJTehrani/PCLA/issues?q=is%3Aissue+is%3Aclosed).

If you have a request for a new agent to be integrated into the framework, please feel free to open a new issue and ask!

----------

## 8. Citation

If you find **PCLA** useful in your research or project, please consider giving it a star ⭐ and citing our published work.
```bibtex
@inproceedings{tehrani2025pcla,
  title={PCLA: A Framework for Testing Autonomous Agents in the CARLA Simulator},
  author={Tehrani, Masoud Jamshidiyan and Kim, Jinhan and Tonella, Paolo},
  booktitle={Proceedings of the 33rd ACM International Conference on the Foundations of Software Engineering},
  pages={1040--1044},
  year={2025}
}
```
