"""Download pretrained agent weights from the PCLA Hugging Face dataset.

Usage:
    python pcla_functions/download_weights.py                       # everything (default, matches old behaviour)
    python pcla_functions/download_weights.py --all
    python pcla_functions/download_weights.py --agents minddrive orion tt
    python pcla_functions/download_weights.py --list

`--agents` takes agent *base* names -- the top-level keys in agents.json
(e.g. "tfv4", "minddrive"), not variants (e.g. not "minddrive_05b" or "tfv4_lav").

Each base name maps to one or more `pcla_agents/<name>_pretrained/` folders,
downloaded via `huggingface_hub.snapshot_download` so large folders resume
instead of restarting, and so downloading a subset only transfers that subset
(the weights are hosted as individual per-agent folders, not one archive).
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

HF_REPO_ID = "MasoudJTehrani/PCLA"
HF_REPO_TYPE = "dataset"

# Maps each agents.json top-level ("base") name to the pretrained folder(s) it
# needs under pcla_agents/. NOT always the obvious `<name>_pretrained` -- some
# agents resolve their checkpoint path indirectly through a config file (lav,
# lmdrive, if, and carl's "plant" variant), and lbc/wor share one folder.
# Keep in sync with agents.json when adding a new agent.
AGENT_PRETRAINED_FOLDERS = {
    "orion":     ["orion_pretrained"],
    "minddrive": ["minddrive_pretrained"],
    "tt":        ["tt_pretrained"],
    "autoware":  [],  # live ROS2/docker bridge -- no weights to download
    "plant2":    ["plant2_pretrained"],
    "tfv3":      ["transfuserv3_pretrained"],
    "tfv4":      ["transfuserv4_pretrained"],
    "tfv5":      ["transfuserv5_pretrained"],
    "tfv6":      ["transfuserv6_pretrained"],
    # carl/carlv11/roach use carl_pretrained; the "plant" variant under this
    # same family resolves (via config/eval{seed}.yaml) to plant_pretrained.
    "carl":      ["carl_pretrained", "plant_pretrained"],
    "lav":       ["lav_pretrained"],
    "lbc":       ["wor_pretrained"],  # lbc_agent.py loads wor_pretrained checkpoints
    "wor":       ["wor_pretrained"],
    "lmdrive":   ["lmdrive_pretrained"],
    "simlingo":  ["simlingo_pretrained"],
    "neat":      ["neat_pretrained"],
    "if":        ["interfuser_pretrained"],
}


def _pcla_agents_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), "pcla_agents")


def _check_agents_json_sync():
    """Warn (never fail) if agents.json and AGENT_PRETRAINED_FOLDERS have drifted."""
    import json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agents_json_path = os.path.join(os.path.dirname(script_dir), "agents.json")
    try:
        with open(agents_json_path) as f:
            known = set(json.load(f).keys())
    except (OSError, json.JSONDecodeError):
        return
    mapped = set(AGENT_PRETRAINED_FOLDERS.keys())
    missing = known - mapped
    if missing:
        print(f"WARNING: agents.json defines {sorted(missing)} with no entry in "
              f"AGENT_PRETRAINED_FOLDERS (pcla_functions/download_weights.py). "
              f"Their weights won't be fetched by name or by --all until added.")


def list_agents():
    print("Valid --agents names (base name, not variant) and the folder(s) each needs:\n")
    for name in sorted(AGENT_PRETRAINED_FOLDERS):
        folders = AGENT_PRETRAINED_FOLDERS[name]
        print(f"  {name:12s} -> {', '.join(folders) if folders else '(no weights needed)'}")


def resolve_folders(agent_names):
    """Validate requested agent base names and return the deduped folder list."""
    unknown = [n for n in agent_names if n not in AGENT_PRETRAINED_FOLDERS]
    if unknown:
        print(f"ERROR: unknown agent name(s): {unknown}")
        print(f"Valid names: {sorted(AGENT_PRETRAINED_FOLDERS)}")
        print("(These are base agent names, e.g. 'tfv4', not variants like 'tfv4_lav'.)")
        sys.exit(1)
    folders = []
    for n in agent_names:
        for f in AGENT_PRETRAINED_FOLDERS[n]:
            if f not in folders:
                folders.append(f)
    return folders


def download_pretrained_folders(folder_names):
    """Download the given pcla_agents/<folder>/ trees from the HF dataset repo."""
    if not folder_names:
        print("Nothing to download (the requested agent(s) need no separate weights).")
        return
    target_dir = _pcla_agents_dir()
    print(f"Repo       : {HF_REPO_ID} ({HF_REPO_TYPE})")
    print(f"Target dir : {target_dir}")
    print(f"Folders    : {', '.join(folder_names)}\n")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        allow_patterns=[f"{folder}/*" for folder in folder_names],
        local_dir=target_dir,
    )
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Download every agent's pretrained weights.")
    group.add_argument("--agents", nargs="+", metavar="NAME",
                       help="Download weights only for these agent base names (see --list).")
    group.add_argument("--list", action="store_true", help="List valid agent base names and exit.")
    args = parser.parse_args()

    _check_agents_json_sync()

    if args.list:
        list_agents()
        return

    if args.agents:
        folders = resolve_folders(args.agents)
    else:
        # Bare invocation (no flags) defaults to --all, matching the previously
        # documented `python pcla_functions/download_weights.py` usage.
        folders = sorted({f for fs in AGENT_PRETRAINED_FOLDERS.values() for f in fs})

    download_pretrained_folders(folders)


if __name__ == "__main__":
    main()
