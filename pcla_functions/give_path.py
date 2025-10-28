import os
import json
from .print_guide import print_guide

def give_path(name, PCLA_dir, routePath):
    """
    Given the name of the agent, return its path and config path
    Set the environment variables if needed
    """
    
    nameArray = name.split("_")

    if nameArray[0] == "tfpp" or nameArray[1] == "carl" or nameArray[1] == "roach" or nameArray[1] == "plant": 
        if nameArray[1] == 'l6':
            os.environ['UNCERTAINTY_THRESHOLD'] = '0.33'
        elif nameArray[1] == 'lav':
            os.environ['STOP_CONTROL'] = '1'
        elif nameArray[1] == 'wp' or nameArray[1] == 'aim':
            os.environ['DIRECT'] = '0'
        elif nameArray[1] == 'roach':
            os.environ['SAMPLE_TYPE'] = 'roach'
    else:
        nameArray.append("")
        if nameArray[0] == "if" or nameArray[0] == "lmdrive":
            os.environ['ROUTES'] = routePath

    # open json file
    with open(os.path.join(PCLA_dir, "agents.json"), 'r') as file:
        agentsFile = json.load(file)
        
        # get agent and its config path
        try:
            agent = agentsFile[nameArray[0]][nameArray[1]]["agent"]
            config_path = agentsFile[nameArray[0]][nameArray[1]]["config"]
            # Split the config path to handle file extension to handle seeds
            config_base, config_ext = os.path.splitext(config_path)
            config = config_base + nameArray[2] + config_ext
        except KeyError:
            print("Couldn't find your model")
            print_guide()

    return os.path.join(PCLA_dir, agent), os.path.join(PCLA_dir, config)
