#!/bin/bash
#SBATCH --job-name=Roach_004
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=28
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/CaRL/CARLA/results/logs/Roach_004_%a_%A.out
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/CaRL/CARLA/results/logs/Roach_004_%a_%A.err
#SBATCH --partition=a100-fat-galvani

start=`date +%s`
echo "START TIME: $(date)"
export SCENARIO_RUNNER_ROOT=/mnt/lustre/work/geiger/bjaeger25/CaRL/CARLA/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/mnt/lustre/work/geiger/bjaeger25/CaRL/CARLA/custom_leaderboard/leaderboard
export CARLA_ROOT=/mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}


repetition=$(($SLURM_ARRAY_TASK_ID))
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "Roach_%02d" ${repetition})
python -u train_parallel.py --train_cpp 0 --team_code_folder /mnt/lustre/work/geiger/bjaeger25/CaRL/CARLA/team_code --ml_cloud 1 --use_traj_sync_ppo False --num_nodes 1 --node_id 0 --rdzv_addr 127.0.0.1 --rdzv_port 0 --collect_device gpu --train_device gpu --PYTORCH_KERNEL_CACHE_PATH /mnt/lustre/work/geiger/bjaeger25/home/.cache --ppo_cpp_install_path /mnt/lustre/work/geiger/bjaeger25/ppo.cpp/install/bin --cpp_singularity_file_path /mnt/lustre/work/geiger/bjaeger25/ppo.cpp/tools/ppo_cpp.sif --git_root /mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla --cpp_system_lib_path_1 /usr/local/cuda-11.8/targets/x86_64-linux/lib --carla_root /mnt/lustre/work/geiger/bjaeger25/CARLA_0_9_15 --exp_name "${ex_name}" --use_dd_ppo_preempt 0 --num_envs_per_gpu 6 --seed ${program_seed} --start_port ${start_port} --gpu_ids 0 --train_towns 1 2 3 4 5 6 --num_envs_per_node 6 --total_batch_size 12288 --total_minibatch_size 256 --norm_adv 0 --clip_vloss 0 --update_epochs 20 --ent_coef 0.01 --vf_coef 0.5 --gamma 0.99 --gae_lambda 0.9 --clip_coef 0.2 --max_grad_norm 0.5 --learning_rate 0.00001 --compile_model False --cpu_collect False --total_timesteps 10000000 --use_exploration_suggest True --lr_schedule kl --use_speed_limit_as_max_speed 0 --beta_min_a_b_value 0.0 --use_new_bev_obs 0 --reward_type roach --consider_tl 1 --eval_time 1200 --terminal_reward 0.0 --normalize_rewards 0 --speeding_infraction 0 --min_thresh_lat_dist 3.5 --map_folder maps_low_res --pixels_per_meter 5 --route_width 16 --num_route_points_rendered 80 --use_green_wave 0 --image_encoder roach --use_comfort_infraction 0 --use_layer_norm 0 --use_vehicle_close_penalty 0 --routes_folder 1000_meters_old_scenarios_01 --render_green_tl 1 --distribution beta --use_termination_hint 0 --use_perc_progress 0 --use_min_speed_infraction 0 --use_leave_route_done 1 --use_temperature False --use_rpo False --rpo_alpha 0.5 --use_layer_norm_policy_head 0 --obs_num_measurements 8 --use_extra_control_inputs 0 --use_hl_gauss_value_loss False --condition_outside_junction 1 --use_outside_route_lanes 0 --use_max_change_penalty 0 --terminal_hint 10.0 --use_lstm False --penalize_yellow_light 0 --use_target_point 0 --use_value_measurements 0 --bev_semantics_width 192 --bev_semantics_height 192 --pixels_ev_to_bottom 40 --use_history 1 --obs_num_channels 15 --use_off_road_term 0 --beta_1 0.9 --beta_2 0.999 --track True --route_repetitions 100 --render_shoulder 1 --use_shoulder_channel 0 --lane_distance_violation_threshold 0.0 --lane_dist_penalty_softener 1.0 --comfort_penalty_factor 0.5 --use_survival_reward 0 &
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60)) 2>&1 | tee /mnt/lustre/work/geiger/bjaeger25/CaRL/CARLA/results/"${ex_name}"/train_time.txt
