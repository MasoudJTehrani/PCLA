#!/bin/bash

export git_root=$1
export num_envs=$2
export num_nodes=$3
export rdzv_addr=$4
export rdzv_port=$5

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8 # TODO tune
export MASTER_ADDR=${rdzv_addr}
#export NCCL_BLOCKING_WAIT=1 # Experimental for debugging.
#export CUDA_LAUNCH_BLOCKING=1
torchrun --start-method spawn --nproc_per_node=${num_envs} --nnodes=${num_nodes} --max_restarts=0 --rdzv-backend=c10d --rdzv-endpoint=${rdzv_addr}:${rdzv_port} ${git_root}/team_code/dd_ppo.py "${@:6}"