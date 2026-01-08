import os
import json
from .print_guide import print_guide

def give_path(name, PCLA_dir, routePath):
    """
    Given the name of the agent, return its path and config path.
    Handles optional variant/seed suffixes and sets environment variables when needed.
    """

    nameArray = name.split("_")
    agent_name = nameArray[0] if nameArray else ""
    variant = nameArray[1] if len(nameArray) > 1 else ""
    seed_suffix = nameArray[2] if len(nameArray) > 2 else ""

    if agent_name == "tfv6":
        # Limit thread usage for TensorFlow-based agents
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'

    elif agent_name == "tfv4" or variant in ("carl", "roach", "plant"):
        # Clear tfv4-specific env vars to prevent cross-contamination between variants
        os.environ.pop('UNCERTAINTY_THRESHOLD', None)
        os.environ.pop('STOP_CONTROL', None)
        os.environ.pop('DIRECT', None)
        
        if variant == 'l6':
            os.environ['UNCERTAINTY_THRESHOLD'] = '0.33'
        elif variant == 'lav':
            os.environ['STOP_CONTROL'] = '1'
        elif variant == 'wp' or variant == 'aim':
            os.environ['DIRECT'] = '0'
        elif variant == 'roach':
            os.environ['SAMPLE_TYPE'] = 'roach'
    else:
        if agent_name == "if" or agent_name == "lmdrive":
            os.environ['ROUTES'] = routePath

    # open json file
    with open(os.path.join(PCLA_dir, "agents.json"), 'r') as file:
        agentsFile = json.load(file)

        agent_entry = agentsFile.get(agent_name)
        if agent_entry is None:
            print("Couldn't find your agent name; please use the format <agent>_<variant>[_seed].")
            print_guide()
            raise ValueError(f"Unknown agent name '{agent_name}'")

        # If no variant was provided, default to the first available for that agent name
        if not variant:
            variant = next(iter(agent_entry.keys()))
        elif variant not in agent_entry:
            print(f"Couldn't find your model variant '{variant}' for agent '{agent_name}'.")
            print_guide()
            raise ValueError(f"Unknown model variant '{variant}' for agent '{agent_name}'")

        # get agent and its config path
        agent = agent_entry[variant]["agent"]
        config_path = agent_entry[variant]["config"]
        # Split the config path to handle file extension and optional seeds
        config_base, config_ext = os.path.splitext(config_path)
        config = config_base + seed_suffix + config_ext

    return os.path.join(PCLA_dir, agent), os.path.join(PCLA_dir, config)
