import os
import subprocess
import time
import json
import argparse
from tqdm.autonotebook import tqdm

# You will likely have to customize this function a bit to work with your cluster partition names etc.
def bash_file(job, carla_world_port_start, carla_streaming_port_start, carla_tm_port_start):
    agent_config = job["agent_config"]
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    # viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]
    with open(job_file, 'w', encoding='utf-8') as rsh:
            rsh.write(f'''#!/bin/bash
#SBATCH --job-name=PlanT_{seed}_{route_id}
#SBATCH --partition=day
#SBATCH -o {log_file}
#SBATCH -e {err_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32gb
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1080ti:1

echo JOB ID $SLURM_JOB_ID

# source ~/.bashrc
# . $CONDA_ROOT/etc/profile.d/conda.sh # idk why i need to do this, bashrc should be enough
# conda activate PlanTUpdate
# cd {"cfggarageroot"}

# export PLANT_VIZ={"viz_path"}
# export PLANT_CHECKPOINT={"cfgcheckpoint"}
# export PYTHONPATH=/home/gerstenecker/PlanTUpdate/leaderboard:/home/gerstenecker/PlanTUpdate/scenario_runner:/home/gerstenecker/garage_2_cleanup/carla/PythonAPI/carla/
# export CARLA_ROOT={"cfgcarla_root"}
# export SCENARIO_RUNNER_ROOT=/home/gerstenecker/PlanTUpdate/scenario_runner
# export LEADERBOARD_ROOT=/home/gerstenecker/PlanTUpdate/leaderboard

FREE_WORLD_PORT=`comm -23 <(seq {carla_world_port_start} {carla_world_port_start + 49} | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
echo 'World Port:' $FREE_WORLD_PORT

FREE_STREAMING_PORT=`comm -23 <(seq {carla_streaming_port_start} {carla_streaming_port_start + 49} | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
echo 'Streaming Port:' $FREE_STREAMING_PORT

export TM_PORT=`comm -23 <(seq {carla_tm_port_start} {carla_tm_port_start+49} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT


${{CARLA_ROOT}}/CarlaUE4.sh -carla-rpc-port=${{FREE_WORLD_PORT}} -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 -carla-streaming-port=${{FREE_STREAMING_PORT}} &

sleep 60  # Wait for CARLA to finish starting

export PORT=${{FREE_WORLD_PORT}}

python -u ${{LEADERBOARD_ROOT}}/leaderboard/leaderboard_evaluator.py --routes={route} \
--repetitions=1 \
--track=MAP \
--checkpoint={result_file} \
--timeout=300 \
--agent=PlanT_agent.py \
--agent-config={agent_config} \
--port=${{FREE_WORLD_PORT}} \
--traffic-manager-port=${{TM_PORT}} \
--traffic-manager-seed={seed}
''')

def get_running_jobs():
    running_jobs = subprocess.check_output(f'squeue --me',shell=True).decode('utf-8').splitlines()
    running_jobs = set(x.strip().split(" ")[0] for x in running_jobs[1:])
    return running_jobs

def filter_completed(jobs):
    filtered_jobs = []

    running_jobs = get_running_jobs()
    for job in jobs:

        # If job is running we keep it in list (other function does killing)
        if "job_id" in job:
           if job["job_id"] in running_jobs:
              filtered_jobs.append(job)
              continue

        # Keep failed jobs to resubmit
        result_file = job["result_file"]
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    evaluation_data = json.load(f)
            except:
                if job["tries"] > 0:
                    filtered_jobs.append(job)
                continue

            progress = evaluation_data['_checkpoint']['progress']

            need_to_resubmit = False
            if len(progress) < 2 or progress[0] < progress[1] or len(evaluation_data['_checkpoint']['records']) == 0:
                need_to_resubmit = True
            else:
                for record in evaluation_data['_checkpoint']['records']:
                    if record['status'] == 'Failed - Agent couldn\'t be set up':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed - Simulation crashed':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed - Agent crashed':
                        need_to_resubmit = True

            if need_to_resubmit and job["tries"] > 0:
                filtered_jobs.append(job)
        # Results file doesnt exist
        elif job["tries"] > 0:
            filtered_jobs.append(job)
    return filtered_jobs

def kill_dead_jobs(jobs):

    running_jobs = get_running_jobs()

    for job in jobs:

        if "job_id" in job:
            job_id = job["job_id"]

        elif os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                job_id = f.readline().strip().replace("JOB ID ", "")
        
        else:
            continue

        if job_id not in running_jobs:
            continue

        log = job["log_file"]
        if not os.path.exists(job["log_file"]):
            continue

        with open(log) as f:
            lines = f.readlines()
        if len(lines)==0:
            continue

        if any(["Watchdog exception" in line for line in lines]) or \
            "Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.\n" in lines or \
            "[91mStopping the route, the agent has crashed:\n" in lines or \
            "[91mError during the simulation:\n" in lines:

            subprocess.Popen(f"scancel {job_id}", shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str, default='data/longest6_split',
                      help='Path to folder containing the split route files')
    parser.add_argument('--config', type=str, default='config/eval.yaml',
                      help='Path to agent config file')
    parser.add_argument('--out_root', type=str, default='results/longest6',
                      help='Path where results should be stored')
    parser.add_argument('--seeds', nargs='+', type=str, default=["1", "2", "3"],
                      help='The seeds to evaluate')
    parser.add_argument('--retries', nargs='+', type=int, default='3',
                      help='Maximum number of retries per route')

    args, unknown = parser.parse_known_args()

    routes = [x for x in os.listdir(args.routes) if x[-4:]==".xml"]

    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    seeds = args.seeds
    retries = args.retries
    agent_config = args.config

    # Filling the job queue
    job_queue = []
    for seed in seeds:

        base_dir = os.path.join(out_root, seed)
        os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)

        for route in routes:
            route_id = route.split("_")[-1][:-4]
            route = os.path.join(args.routes, route)

            result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
            log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
            err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")

            job_file = os.path.join(base_dir, "run", f'eval_{route_id}.sh')
            
            job = {
                "agent_config": agent_config,
                "route": route,
                "route_id": route_id,
                "seed": seed,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "job_file": job_file,
                "tries": retries
            }

            job_queue.append(job)

    carla_world_ports = list(range(10000, 20000, 50))
    carla_streaming_ports = list(range(20000, 30000, 50))
    carla_tm_ports = list(range(30000, 40000, 50))
    port_idx = 0

    # Submitting the jobs to slurm
    jobs = len(job_queue)
    progress = tqdm(total = jobs)
    while job_queue:
        kill_dead_jobs(job_queue)
        job_queue = filter_completed(job_queue)

        progress.update(jobs - len(job_queue) - progress.n)

        running_jobs = get_running_jobs()

        used_ports = set()
        for job in job_queue:
            if "job_id" in job and job["job_id"] in running_jobs:
                used_ports.update(job["ports"])

        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())

        if len(running_jobs) >= max_num_parallel_jobs:
            time.sleep(5)
            continue

        for job in job_queue:
            if job["tries"] <= 0:
                continue

            if "job_id" in job and job["job_id"] in running_jobs:
                continue

            if os.path.exists(job["log_file"]):
                with open(job["log_file"], "r") as f:
                    job_id = f.readline().strip().replace("JOB ID ", "")
                    if job_id in running_jobs:
                        continue

            # Need to submit this job
            carla_world_port_start = carla_world_ports[port_idx]
            carla_streaming_port_start = carla_streaming_ports[port_idx]
            carla_tm_port_start = carla_tm_ports[port_idx]
            port_idx += 1

            # Make bash file:
            bash_file(job, carla_world_port_start, carla_streaming_port_start, carla_tm_port_start)
            job["ports"] = {carla_world_port_start, carla_streaming_port_start, carla_tm_port_start}

            # submit
            for file in [job["result_file"], job["err_file"], job["log_file"]]:
                if os.path.exists(file):
                    os.remove(file)
            job_id = subprocess.check_output(f'sbatch {job["job_file"]}', shell=True).decode('utf-8').strip().rsplit(' ', maxsplit=1)[-1]
            
            job["job_id"] = job_id
            job["tries"] -= 1

            print(f'submit {job["job_file"]} {job_id}')
            print(len(job_queue))
            break

        time.sleep(2)