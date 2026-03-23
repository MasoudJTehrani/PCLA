#!/bin/bash
#SBATCH --job-name=plant_train
#SBATCH -o train/train_plant_final_3_out.log
#SBATCH -e train/train_plant_final_3_err.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=2080-galvani
#SBATCH --time=18:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8

export SEED=3
export CHECKPOINT_ADDON="final"
export DS_LOCAL=/scratch_local/$(ls /scratch_local | grep gwb301 | tail -n 1)/tmp/$(openssl rand -hex 2)

export DS=$DS_LOCAL/plant_dataset

mkdir -p $DS

echo $DS

# This is specifically for the slurm cluster so the dataset is on the ssd
date
unzip -q ~/data/PlanT_dataset.zip -d $DS
date

python -u lit_train.py