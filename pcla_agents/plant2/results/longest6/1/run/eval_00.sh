#!/bin/bash
#SBATCH --job-name=PlanT_1_00
#SBATCH --partition=day
#SBATCH -o results/longest6/1/out/00_out.log
#SBATCH -e results/longest6/1/err/00_err.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50gb
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1080ti:1

echo JOB ID $SLURM_JOB_ID

# source ~/.bashrc
# . $CONDA_ROOT/etc/profile.d/conda.sh # idk why i need to do this, bashrc should be enough
# conda activate PlanTUpdate
# cd cfggarageroot

export PLANT_VIZ=results/longest6/1/viz/00
export PLANT_CHECKPOINT=/home/gerstenecker/plant2/checkpoints/epoch=029_final_1.ckpt
export PYTHONPATH=/home/gerstenecker/garage_2_cleanup/carla/PythonAPI:/home/gerstenecker/garage_2_cleanup/carla/PythonAPI/carla:/home/gerstenecker/PlanT_2_cleanup/scenario_runner_autopilot:/home/gerstenecker/PlanT_2_cleanup/leaderboard_autopilot:/home/gerstenecker/PlanT_2_cleanup/PlanT:/home/gerstenecker/PlanT_2_cleanup/carla_garage
export CARLA_ROOT=/home/gerstenecker/garage_2_cleanup/carla
export SCENARIO_RUNNER_ROOT=/home/gerstenecker/PlanT_2_cleanup/scenario_runner_autopilot
export LEADERBOARD_ROOT=/home/gerstenecker/PlanT_2_cleanup/leaderboard_autopilot
export WORK_DIR=/home/gerstenecker/PlanT_2_cleanup
export IS_BENCH2DRIVE=0

FREE_WORLD_PORT=`comm -23 <(seq 18500 18549 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'World Port:' $FREE_WORLD_PORT

FREE_STREAMING_PORT=`comm -23 <(seq 22650 22699 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'Streaming Port:' $FREE_STREAMING_PORT

export TM_PORT=`comm -23 <(seq 34500 34549 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT

# # TODO ############################################ with or without graphics?

if [ ${IS_BENCH2DRIVE} -eq 0 ]
then
    ${CARLA_ROOT}/CarlaUE4.sh -carla-rpc-port=${FREE_WORLD_PORT} -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 -carla-streaming-port=${FREE_STREAMING_PORT} &
    sleep 60  # Wait for CARLA to finish starting
fi

python -u /home/gerstenecker/PlanT_2_cleanup/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py --routes=/home/gerstenecker/garage_2_cleanup/leaderboard/data/longest6_split/longest6_00.xml --repetitions=1 --track=MAP --checkpoint=results/longest6/1/res/00_res.json --timeout=300 --agent=PlanT_agent.py --agent-config= --port=${FREE_WORLD_PORT} --traffic-manager-port=${TM_PORT} --traffic-manager-seed=1 
