<p align="center" style="font-size:40px;">
  <b>PCLA: A framework for testing autonomous agents in the CARLA simulator</b>
  <br>
  <br>
  PCLA agents:
  <br>
  <br>
  <b>--- SimLingo (CarLLava), LMDrive, Transfuser, Transfuser++, Interfuser, NEAT, World on Rails (WoR), Learning By Cheating (LBC), Learning from All Vehicles (LAV) ---</b>
</p>

---

<p align="center">
PCLA (Pretrained CARLA Leaderboard Agents) is a versatile framework that allows you to use and evaluate the autonomous agents from the <a href="https://leaderboard.carla.org">CARLA Leaderboard</a> independently of its core codebase and put them on your vehicle. </br>

* PCLA provides a clear method to deploy Autonomous Driving Agents (ADAs) onto a vehicle without relying on the Leaderboard codebase.
* Enables easy switching between ADAs without requiring changes to CARLA versions or programming environments.
* Allows you to have multiple vehicles with different autonomous agents (requires high graphical memory).
* Provides the next movement action computed by the chosen agent, which can then be used in any desired application.
* Is fully compatible with the latest version of CARLA and independent of the Leaderboard’s specific CARLA version.
* Includes **23** different high-performing ADAs trained with 31 distinct training seeds. 

Paper available at <a href="https://dl.acm.org/doi/abs/10.1145/3696630.3728577">Foundations of Software Engineering</a>

</p>

<p align="center">
<strong>PCLA was tested on Linux Ubuntu 22 and CARLA 0.9.16 Unreal Engine 4.</strong> </br>
A video tutorial on how to use PCLA is available below (update will come soon).
  
<div align="center">
  <a href="https://www.youtube.com/watch?v=QyaMK6vclBg"><img src="https://img.youtube.com/vi/QyaMK6vclBg/0.jpg" alt="PCLA Video Tutorial"></a>
</div>

</p>


## Contents

1. [Setup](#setup)
2. [Pre-Trained Weights](#pre-trained-weights)
3. [Autonomous Agents](#autonomous-agents)
4. [How to Use](#how-to-use)
5. [Navigation](#navigation)
6. [Sample Code](#sample-code)
7. [FAQ](#FAQ)
8. [Citation](#citation)

## Setup
Download and install the <a href="https://carla.readthedocs.io/en/latest/">CARLA simulator</a> from the official website. Based on your preference, you can either use quick installation or build from source.</br>
Please make sure CUDA and PyTorch are installed.</br>
<a href="https://www.gpu-mart.com/blog/install-nvidia-cuda-11-on-ubuntu">Tutorial for installing CUDA on ubuntu<a></br>
<a href="https://pytorch.org/get-started/locally/">Tutorial for PyTorch<a>

Clone the repository and build the conda environment:

```Shell
git clone https://github.com/MasoudJTehrani/PCLA
cd PCLA
conda env create -f environment.yml
conda activate PCLA
```
Please make sure to install **torch-scatter** according to your own CUDA version. You can check your CUDA version using the `python cuda.py` code.

--If you want to use PCLA for **CARLA 0.9.16**, you need to install the wheel from the `dist` folder.

```Shell
cd dist
python3 -m pip install carla-0.9.16-cp38-cp38-linux_x86_64.whl
```
**Note**: Some agents act _weird_ in CARLA 0.9.16

Additional setups for the LMDrive agent:
```Shell
conda activate PCLA

cd plca_agents/lmdrive/vision_encoder
pip uninstall timm
python setup.py develop

cd ../LAVIS
python setup.py develop

pip uninstall ftfy -y
pip install "ftfy==6.1.1"
```

## Pre-Trained Weights

You can either run this code to download and unzip the weights automatically
```bash
python download_weights.py
``` 
or manually download the pre-trained weights from <a href="https://zenodo.org/records/17399201">Zenodo</a> or directly from <a href="https://zenodo.org/records/17399201/files/pretrained.zip?download=1">here</a> and extract them into the `PCLA/agents/` directory.</br>
Ensure that each folder of pre-trained weights is placed directly next to its respective model's folder. The `agents` folder should look like this.
```bash
├── agents
   ├── transfuserpp
   ├── transfuserpp_pretrained
   ├── interfuser
   ├── interfuser_pretrained
   ├── ...
```

## Autonomous Agents

PCLA includes 23 different autonomous agents and 31 distinct training seeds to choose from.
- **SimLingo(CarLLava)**
  - Contains 1 agent from the leaderboard 2, previously named CarLLava.
    - **simlingo_simlingo** : The best performing agent, first place at <a href="https://leaderboard.carla.org">CARLA Leaderboard 2</a> SENSORS track.
  - <a href="https://github.com/RenzKa/simlingo">Repository</a>

- **LMDrive**
  - Contains 3 agent from the leaderboard 1.
    - **lmdrive_llava** : The best performing LMDrive agent.
    - **lmdrive_vicuna** : The second best performing LMDrive agent.
    - **lmdrive_llama** : The third best performing LMDrive agent.
  - <a href="https://github.com/opendilab/LMDrive">Repository</a>

- **Transfuser**
  - Contains 4 agent from the leaderboard 1 transfuser.
    - **tf_tf** : The main Transfuser agent.
    - **tf_ltf** : The LatentTF agent.
    - **tf_lf** : The Late_Fusion agent.
    - **tf_gf** : The Geometric_Fusion agent.
  - <a href="https://github.com/autonomousvision/transfuser">Repository</a>
  
- **Transfuser++**
  - Contains 4 different autonomous agents of Transfuser++ with 3 training seeds for each agent.
    - **tfpp_l6_#** : Best performing Transfuser++ agent. Second place at <a href="https://leaderboard.carla.org">CARLA Leaderboard 2</a> SENSORS track(Tuebingen_AI team)
    - **tfpp_lav_#** : Transfuser++ but it's not trained on Town02 and Town05.
    - **tfpp_wp_#** : Transfuser++ WP from their paper's appendix.
    - **tfpp_aim_#** : Reproduction of the <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Prakash_Multi-Modal_Fusion_Transformer_for_End-to-End_Autonomous_Driving_CVPR_2021_paper.html" target="_blank">AIM </a>method, explained in their paper's appendix.

  - Replace # with the seed number from 0 to 2.
  - <a href="https://github.com/autonomousvision/carla_garage/tree/leaderboard_1">Repository</a>
    
- **Learning from All Vehicles**
  - Contains 2 autonomous agents. Needs the CARLA to be run with -vulkan.
    - **lav_lav** : The original LAV agent.
    - **lav_fast** : The leaderboard submission of LAV. The codes are slightly optimized for leaderboard inference speed with temporal LiDAR scans.
  - <a href="https://github.com/dotchen/LAV">Repository</a>
    
- **Learning By Cheating**
  - Contains 2 autonomous agents. Needs the CARLA to be run with -vulkan.
    - **lbc_nc** : Learning By Cheating, the NoCrash model.
    - **lbc_lb** : Learning By Cheating, the Leaderboard model.
  - <a href="https://github.com/dotchen/WorldOnRails">Repository</a>
    
- **World on Rails**
  - Contains 2 autonomous agents. Needs the CARLA to be run with -vulkan.
    - **wor_nc** : World on Rails, the NoCrash model.
    - **wor_lb** : World on Rails, the Leaderboard model.
  - <a href="https://github.com/dotchen/WorldOnRails">Repository</a>
    
- **NEAT**
  - Contains 4 different autonomous agents.
      - **neat_neat**
      - **neat_aimbev**
      - **neat_aim2dsem**
      - **neat_aim2ddepth**
  - <a href="https://github.com/autonomousvision/neat">Repository</a>
    
- **Interfuser**
  - Contains 1 autonomous agent.
     - **if_if** : Second best performing <a href="https://leaderboard.carla.org">CARLA Leaderboard 1</a> SENSORS track agent.
  - <a href="https://github.com/opendilab/InterFuser">Repository</a>

## How to Use
First, run CARLA. You **only** need -vulkan for LBC, WoR, and LAV agents
```Shell
./CarlaUE4.sh -vulkan
```
Then open another terminal and run your code.</br>
To use PCLA, simply import it and use the PCLA class to define an autonomous vehicle with your chosen autonomous agent.
```python
from PCLA import PCLA

agent = "tf_tf"
route = "./sample_route.xml"
pcla = PCLA(agent, vehicle, route, client)

ego_action = pcla.get_action()
vehicle.apply_control(ego_action)
```
In the code above, the agent is your chosen autonomous agent. You can choose your agent from the list of [Autonomous Agents](#autonomous-agents).</br>
You also need to pass the `route` that you want your vehicle to follow. The route should be in the format of the Leaderboard waypoints as an `XML` file.</br>
To make it easy, PCLA provides you with a function called `routeMaker()` that gets an array of <a href="https://carla.readthedocs.io/en/latest/core_map/#waypoints" target="_blank">CARLA waypoints</a>, reformats it to a Leaderboard format, and save it as an XML file. A tutorial on how to use that is provided in [Navigation](#navigation)</br>
The other arguments you have to pass to PCLA are the client and the vehicle you want to put the agent on. </br>
To get one action in a frame from the agent and apply it to your vehicle, you can call the `pcla.get_action` method. </br>
Example:
```python
ego_action = pcla.get_action()
vehicle.apply_control(ego_action)
```
Finally, to destroy and cleanup the vehicle, sensors, and the PCLA variables, you can call
```python
pcla.cleanup()
```
## Navigation
You can use PCLA to generate waypoints between two locations or generate routes usable for PCLA.
If you want to find locations to navigate your vehicle through the city, you can use the `spawn_points.py` file to see all the spawn points and their associated number.
```shell
python pcla_functions/spawn_points.py
```

<hr />

You can then use the `location_to_waypoint()` method to generate waypoints between two CARLA locations. For example:
```python
from PCLA import location_to_waypoint

vehicle_spawn_points = world.get_map().get_spawn_points() # Carla spawn points
startLoc = vehicle_spawn_points[31].location # Start location
endLoc = vehicle_spawn_points[42].location # End location
waypoints = location_to_waypoint(client, startLoc, endLoc) # Returns waypoints between two locations
```


Then pass the waypoints to `route_maker()` to make the XML file usable for PCLA.
```python
PCLA.route_maker(waypoints)
```

<hr />

All together:
Example of generating XML route from a list of <a href="https://carla.readthedocs.io/en/latest/core_map/#waypoints" target="_blank">CARLA waypoints</a>
extracted from two <a href="https://carla.readthedocs.io/en/latest/python_api/#carlalocation" target="_blank">CARLA locations</a>:

```python
from PCLA import route_maker
from PCLA import location_to_waypoint

client = carla.Client('localhost', 2000)
world = client.get_world()

vehicle_spawn_points = world.get_map().get_spawn_points() # Carla spawn points
startLoc = vehicle_spawn_points[31].location # Start location
endLoc = vehicle_spawn_points[42].location # End location
waypoints = location_to_waypoint(client, startLoc, endLoc)  # Returns waypoints between two locations
route_maker(waypoints, "route.xml")  # Returns waypoints usable for PCLA
```

## Sample Code
A sample code is provided for you to test PCLA. Just go to the PCLA directory and run:
```Shell
python sample.py
```
This sample is in Town02 of the CARLA simulator and uses the LAV agent.

***Attention: you may need to change the vehicle spawn point's number on line 43 to something else based on your CARLA version.***

## FAQ
Frequently asked questions and possible issues are solved in <a href="https://github.com/MasoudJTehrani/PCLA/issues?q=is%3Aissue+is%3Aclosed" target="_blank">the issues section</a>.
If you have a request for a new agent, feel free to ask me.

## Citation
If you find PCLA useful, please consider giving it a star &#127775;, and cite the published <a href="https://dl.acm.org/doi/abs/10.1145/3696630.3728577">paper</a>.

```bibtex
@inproceedings{tehrani2025pcla,
  title={PCLA: A Framework for Testing Autonomous Agents in the CARLA Simulator},
  author={Tehrani, Masoud Jamshidiyan and Kim, Jinhan and Tonella, Paolo},
  booktitle={Proceedings of the 33rd ACM International Conference on the Foundations of Software Engineering},
  pages={1040--1044},
  year={2025}
}
```
