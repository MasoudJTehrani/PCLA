#!/bin/bash
#SBATCH --job-name=PPO_037
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=14
#SBATCH --output=/weka/geiger/bjaeger25/CaRL/CARLA/results/logs/AC_PPO_037_%a_%A.out
#SBATCH --error=/weka/geiger/bjaeger25/CaRL/CARLA/results/logs/AC_PPO_037_%a_%A.err
#SBATCH --partition=h100-ferranti

start=`date +%s`
echo "START TIME: $(date)"
export SCENARIO_RUNNER_ROOT=/weka/geiger/bjaeger25/CaRL/CARLA/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/weka/geiger/bjaeger25/CaRL/CARLA/custom_leaderboard/leaderboard
export CARLA_ROOT=/weka/geiger/bjaeger25/carla_0_9_15
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}


repetition=$(($SLURM_ARRAY_TASK_ID))
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "CaRL_10M_%02d" ${repetition})
python -u train_parallel.py --train_cpp 1 --team_code_folder /weka/geiger/bjaeger25/CaRL/CARLA/team_code --ml_cloud 1 --num_nodes 1 --node_id 0 --rdzv_addr 127.0.0.1 --rdzv_port 0 --collect_device gpu --train_device gpu --PYTORCH_KERNEL_CACHE_PATH /weka/geiger/bjaeger25/.cache --ppo_cpp_install_path /weka/geiger/bjaeger25/ppo.cpp/install/bin --cpp_singularity_file_path /weka/geiger/bjaeger25/ppo.cpp/tools/ppo_cpp.sif --git_root /weka/geiger/bjaeger25/ad_planning/2_carla --cpp_system_lib_path_1 /weka/geiger/bjaeger25 --cpp_system_lib_path_2 /weka/geiger/bjaeger25/ad_planning/2_carla/results --carla_root /weka/geiger/bjaeger25/carla_0_9_15 --exp_name "${ex_name}" --use_dd_ppo_preempt 0 --num_envs_per_gpu 8 --seed ${program_seed} --start_port ${start_port} --gpu_ids 0 --train_towns 1 2 3 4 5 6 7 10 --num_envs_per_node 8 --num_steps 512 --num_minibatches 4 --norm_adv 1 --clip_vloss 1 --update_epochs 3 --ent_coef 0.01 --vf_coef 0.5 --gamma 0.99 --gae_lambda 0.95 --clip_coef 0.1 --max_grad_norm 0.5 --learning_rate 0.00025 --total_timesteps 10000000 --lr_schedule linear --use_speed_limit_as_max_speed 0 --beta_min_a_b_value 1.0 --use_new_bev_obs 1 --reward_type simple_reward --consider_tl 1 --eval_time 1200 --terminal_reward 0.0 --normalize_rewards 0 --speeding_infraction 1 --min_thresh_lat_dist 2.0 --map_folder maps_2ppm_cv --pixels_per_meter 2 --route_width 6 --num_route_points_rendered 150 --use_green_wave 0 --image_encoder roach_ln2 --use_layer_norm 1 --use_vehicle_close_penalty 0 --routes_folder 1000_meters_old_scenarios_01 --render_green_tl 1 --distribution beta --use_termination_hint 1 --use_perc_progress 1 --use_min_speed_infraction 0 --use_leave_route_done 0 --use_layer_norm_policy_head 1 --obs_num_measurements 8 --use_extra_control_inputs 0 --condition_outside_junction 0 --use_outside_route_lanes 1 --use_max_change_penalty 0 --terminal_hint 1.0 --use_target_point 0 --use_value_measurements 1 --bev_semantics_width 256 --bev_semantics_height 256 --pixels_ev_to_bottom 100 --use_history 0 --obs_num_channels 10 --use_off_road_term 1 --beta_1 0.9 --beta_2 0.999 --route_repetitions 20  --render_speed_lines 1 --use_new_stop_sign_detector 1 --use_positional_encoding 0 --use_ttc 1 --num_value_measurements 10 --render_yellow_time 1 --penalize_yellow_light 0  --use_comfort_infraction 1 --use_single_reward 1 --off_road_term_perc 0.95 --render_shoulder 0 --use_shoulder_channel 1 --use_rl_termination_hint 1 --lane_distance_violation_threshold 0.0 --lane_dist_penalty_softener 1.0 --comfort_penalty_factor 0.5 --use_survival_reward 0 --use_exploration_suggest 0 --torch_deterministic 0 &
wait

end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' $((${runtime}/86400)) $((${runtime}%86400/3600)) $((${runtime}%3600/60)) $((${runtime}%60)) 2>&1 | tee /weka/geiger/bjaeger25/CaRL/CARLA/results/"${ex_name}"/train_time.txt
