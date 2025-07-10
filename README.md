<p align="center" style="font-size:40px;">
<b>PCLA: A framework for testing autonomous agents in the CARLA simulator</b>
</p>

<p align="center">
PCLA (Pretrained CARLA Leaderboard Agent) is a versatile framework that allows you to utilize the autonomous agents from the <a href="https://leaderboard.carla.org/leaderboard/">CARLA Leaderboard</a> independently of its core codebase and put them on your vehicle. </br>

* PCLA provides a clear method to deploy Autonomous Driving Agents (ADAs) onto a vehicle without relying on the Leaderboard codebase.
* Enables easy switching between ADAs without requiring changes to CARLA versions or programming environments.
* Allows you to have multiple vehicles with different autonomous agents.
* Provides the next movement action computed by the chosen agent, which can then be utilized in any desired application.
* Is fully compatible with the latest version of CARLA and independent of the Leaderboard’s specific CARLA version.
* Includes nine different high-performing ADAs trained with 17 distinct training seeds. 

Paper available at <a href="https://arxiv.org/abs/2503.09385">Arxiv</a>. It will be published at the <a href="https://conf.researchr.org/track/fse-2025/fse-2025-demonstrations?#program">FSE 2025 demonstration track<a>.

</p>

<p align="center">
<strong>PCLA was tested on Linux Ubuntu 22, Windows and CARLA 9.15.2 Unreal Engine 4.</strong> </br>
A video tutorial on how to use PCLA is available below.
  
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
6. [Environment Variables](#environment-variables)
7. [Sample Code](#sample-code)
8. [FAQ](#FAQ)
9. [Citation](#citation)

## Setup
Download and install the <a href="https://carla.readthedocs.io/en/latest/">CARLA simulator</a> from the official website. Based on your preference, you can either use quick installation or build from source.</br>
On **Ubuntu**, clone the repository and build the conda environment:
```Shell
git clone https://github.com/MasoudJTehrani/PCLA
cd PCLA
conda env create -f environment.yml
conda activate PCLA
```
Alternatively, you can use the `requirements.txt` file to install the required libraries by using:</br>
```Shell
pip install -r requirements.txt
```
Please make sure CUDA and PyTorch are installed.</br>
<a href="https://www.gpu-mart.com/blog/install-nvidia-cuda-11-on-ubuntu">Tutorial for installing CUDA on ubuntu<a></br>
<a href="https://pytorch.org/get-started/locally/">Tutorial for PyTorch<a>

On **Windows**, follow the same process but now with the `WindowsEnvironment.yml` file forked by <a href="https://github.com/Mutahar789/PCLA/tree/main">Mutahar789</a>

## Pre-Trained Weights

Download the pre-trained weights from <a href="https://zenodo.org/records/15422537">Zenodo</a> or directly from <a href="https://zenodo.org/records/15422537/files/pretrained.zip?download=1">here</a> and extract them into the `PCLA/agents/` directory.</br> 
Ensure that each folder of pre-trained weights is placed directly next to its respective model's folder. The `agents` folder should look like this.
```bash
├── agents
   ├── transfuserpp
   ├── transfuserppPretrained
   ├── interfuser
   ├── interfuserPretrained
   ├── neat
   └── neatPretrained
```

## Autonomous Agents

PCLA includes 9 different autonomous agents and 17 distinct training seeds to choose from.
- **Transfuser++**
  - Contains 4 different autonomous agents of Transfuser++ with 3 training seeds for each agent. To use these agents you need to set some [Environment Variables](#environment-variables).
    - **tfpp_l6_#** : Best performing Transfuser++ agent. Second place at <a href="https://leaderboard.carla.org/leaderboard/">CARLA Leaderboard 2</a> SENSORS track(Tuebingen_AI team)
    - **tfpp_lav_#** : Transfuser++ but it's not trained on Town02 and Town05.
    - **tfpp_wp_#** : Transfuser++ WP from their paper's appendix.
    - **tfpp_aim_#** : Reproduction of the <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Prakash_Multi-Modal_Fusion_Transformer_for_End-to-End_Autonomous_Driving_CVPR_2021_paper.html" target="_blank">AIM </a>method, explained in their paper's appendix.

  - Replace # with the seed number from 0 to 2.
  - Repository: [https://github.com/autonomousvision/carla_garage](https://github.com/autonomousvision/carla_garage)
- **Neat**
  - Contains 4 different autonomous agents. No environment variables are needed for these agents.
      - **neat_neat**
      - **neat_aimbev**
      - **neat_aim2dsem**
      - **neat_aim2ddepth**
  - Repository: [https://github.com/autonomousvision/neat](https://github.com/autonomousvision/neat)
- **Interfuser**
  - Contains 1 autonomous agent. To use this agent you need to set an [Environment Variables](#environment-variables).
     - **if_if** : Second best performing <a href="https://leaderboard.carla.org/leaderboard/">CARLA Leaderboard 1</a> SENSORS track agent.
  - Repository: [https://github.com/opendilab/InterFuser](https://github.com/opendilab/InterFuser)

## How to Use
First, run CARLA. You don't need any special arguments.
```Shell
./CarlaUE4.sh
```
Then open another terminal and run your code.</br>
To use PCLA, simply import it and use the PCLA class to define an autonomous vehicle with your chosen autonomous agent.
```python
from PCLA import PCLA

agent = "neat_neat"
route = "./sampleRoute.xml"
pcla = PCLA(agent, vehicle, route, client)

ego_action = pcla.get_action()
vehicle.apply_control(ego_action)
```
In the code above, the agent is your chosen autonomous agent. You can choose your agent from the list of [Autonomous Agents](#autonomous-agents).</br>
You also need to pass the `route` that you want your vehicle to follow. The route should be in the format of the Leaderboard waypoints as an `XML` file.</br>
To make it easy, PCLA provides you with a function called `routeMaker()` that gets an array of <a href="https://carla.readthedocs.io/en/latest/core_map/#waypoints" target="_blank">CARLA waypoints</a>, reformats it to a Leaderboard format and save it as an XML file. A tutorial on how to use that is provided in [Navigation](#navigation)</br>
The other arguments you have to pass to PCLA are the client, and the vehicle you want to put the agent on. </br>
To get one action in a frame from the agent and apply it to your vehicle you can call the `pcla.get_action` method. </br>
Example:
```python
ego_action = pcla.get_action()
vehicle.apply_control(ego_action)
```
Finally to destroy and cleanup the vehicle, sensors and the PCLA variables you can call
```python
pcla.cleanup()
```
## Navigation
You can use PCLA to generate waypoints between two locations or generate routes usable for PCLA.
If you want to find locations to navigate your vehicle through the city, you can use the `spawn_points.py` file to see all the spawn points and their associated number.
```shell
python spawn_points.py
```

<hr />

You can then use the `location_to_waypoint()` method to generate waypoints between two carla locations. For example:
```python
from PCLA import location_to_waypoint

vehicle_spawn_points = world.get_map().get_spawn_points() # Carla spawn points
startLoc = vehicle_spawn_points[31].location # Start location
endLoc = vehicle_spawn_points[42].location # End location
waypoints = location_to_waypoint(client, startLoc, endLoc) # Returns waypoints between two locations
```

<hr />

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

## Environment Variables
Transfuser++ and Interfuser require you to set an environment variable before using their agents.
Environment variables for each agent are:
- **tfpp_l6_#**
  ```Shell
  export UNCERTAINTY_THRESHOLD=033
  ```
- **tfpp_lav_#**
  ```Shell
  export STOP_CONTROL=1
  ```
- **tfpp_aim_#**
  ```Shell
  export DIRECT=0
  ```
- **tfpp_wp_#**
  ```Shell
  export DIRECT=0
  ```
- **if_if**
  ```Shell
  export ROUTES=path_to_route.xml
  ```
  The path to the same XML route file you used in PCLA

Remember to unset a variable before using another agent.
```Shell
unset DIRECT
```

## Sample Code
A sample code is provided for you to test PCLA. Just go to the PCLA directory and run:
```Shell
python sample.py
```
This sample is in Town02 of the CARLA simulator. 

***Attention: you may need to change the vehicle spawn point's number on line 43 to something else based on your CARLA version.***

## FAQ
Frequently asked questions and possible issues are solved in <a href="https://github.com/MasoudJTehrani/PCLA/issues?q=is%3Aissue+is%3Aclosed" target="_blank">the issues section</a>.
If you have a request for a new agent, feel free to ask me.

## Citation
If you find PCLA useful, please consider giving it a star &#127775;, and cite the published paper:
https://arxiv.org/abs/2503.09385
```bibtex
@article{tehrani2025pcla,
  title={PCLA: A Framework for Testing Autonomous Agents in the CARLA Simulator},
  author={Tehrani, Masoud Jamshidiyan and Kim, Jinhan and Tonella, Paolo},
  journal={arXiv preprint arXiv:2503.09385},
  year={2025}
}
```
