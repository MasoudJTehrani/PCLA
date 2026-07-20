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
        
        if agent_name == "plant2":
            # For visualization purposes, PLANT_VIZ=/path/to/viz_outputs  # set to empty string to disable
            os.environ['PLANT_VIZ'] = ""
            os.environ['PLANT_CHECKPOINT'] = os.path.join(PCLA_dir, config)

        if agent_name == "tt":
            # ThinkTwice's agent setup() expects a single conf string of the form
            # "<checkpoint.pth>+<network_config.py>" (it splits on '+'). `config` from
            # agents.json is the checkpoint path; the mmdet3d network config lives in the
            # staged model tree. Return both as absolute paths joined by '+'.
            tt_ckpt = os.path.join(PCLA_dir, config)
            tt_cfg = os.path.join(PCLA_dir, "pcla_agents/tt/open_loop_training/configs/thinktwice.py")
            return os.path.join(PCLA_dir, agent), tt_ckpt + "+" + tt_cfg

        if agent_name == "minddrive":
            # MindDrive's setup() splits its conf string on '+' and takes
            # "<network_config.py>+<checkpoint.pth>" -- note this is the OPPOSITE
            # order to ThinkTwice above. `config` from agents.json is the checkpoint;
            # the matching PCLA config variant (local llm_path, flash_attn=False)
            # lives in the staged model tree, and differs per model size.
            md_configs = {
                "05b": "minddrive_qwen2_05B_infer_pcla.py",
                "3b": "minddrive_qwen25_3B_infer_pcla.py",
            }
            if variant not in md_configs:
                raise ValueError(
                    f"Unknown minddrive variant '{variant}'. "
                    f"Expected one of: {', '.join(sorted(md_configs))}")
            md_ckpt = os.path.join(PCLA_dir, config)
            md_cfg = os.path.join(
                PCLA_dir, "pcla_agents/minddrive/adzoo/minddrive/configs", md_configs[variant])
            return os.path.join(PCLA_dir, agent), md_cfg + "+" + md_ckpt

        if agent_name == "orion":
            # ORION uses the same "<network_config.py>+<checkpoint.pth>" conf string as
            # minddrive (MindDrive is a fork of ORION). The fp16 config is used
            # deliberately: ORION is a ~7B model and needs fp16 to fit a 20 GB card.
            or_configs = {
                # Derived from `orion_stage3_agent.py` -- the only inference config
                # that defines `inference_only_pipeline`, which the agent requires --
                # plus fp16 inference so ~7.5B params fit a 20 GB card.
                "base": "orion_stage3_agent_fp16_pcla.py",
            }
            if variant not in or_configs:
                raise ValueError(
                    f"Unknown orion variant '{variant}'. "
                    f"Expected one of: {', '.join(sorted(or_configs))}")
            or_ckpt = os.path.join(PCLA_dir, config)
            or_cfg = os.path.join(
                PCLA_dir, "pcla_agents/orion/adzoo/orion/configs", or_configs[variant])
            return os.path.join(PCLA_dir, agent), or_cfg + "+" + or_ckpt

    return os.path.join(PCLA_dir, agent), os.path.join(PCLA_dir, config)
