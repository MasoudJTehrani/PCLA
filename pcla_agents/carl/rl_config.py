'''
Config class that contains all the hyperparameters needed to build any model.
'''
import numpy as np


class GlobalConfig:
  '''
  Config class that contains all the hyperparameters needed to build any model.
  '''

  def __init__(self):
    self.frame_rate = 10.0  # Frames per second of the CARLA simulator
    self.original_frame_rate = 20.0  # Frames per second of the CARLA evaluation server. 20 comes from leaderboard
    self.time_interval = 1.0 / self.frame_rate  # ms per step in CARLA time.

    self.pixels_per_meter = 5.0  # 1 / pixels_per_meter = size of pixel in meters
    self.bev_semantics_width = 192  # Numer of pixels the bev_semantics is wide
    self.pixels_ev_to_bottom = 40  # Numer of pixels from the vehicle to the bottom.
    self.bev_semantics_height = 192  # Numer of pixels the bev_semantics is high
    # Distance of traffic lights considered relevant (in meters)
    self.light_radius = 15.0
    self.debug = False  # Whether to turn on debugging functions, like visualizations.
    self.logging_freq = 10  # Log every 10 th frame
    self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
    self.route_points = 10  # Number of route points to render in logger

    half_second = int(self.frame_rate * 0.5)
    self.history_idx = [-3 * half_second - 1, -2 * half_second - 1, -1 * half_second - 1, -0 * half_second - 1]

    self.num_route_points_rendered = 80  # Number of route points rendered into the BEV seg observation.
    self.use_history = False  # Whether to use the history in bev_observation
    # History index for bev_observation
    # Current observation is not included as it is rendered anyway
    self.history_idx_2 = [-3 * half_second - 1, -2 * half_second - 1, -1 * half_second - 1]

    # Color format BGR
    self.bev_classes_list = (
        (0, 0, 0),  # unlabeled
        (150, 150, 150),  # road
        (255, 255, 255),  # route
        (255, 255, 0),  # lane marking
        (0, 0, 255),  # vehicle
        (0, 255, 255),  # pedestrian
        (255, 255, 0),  # traffic light
        (160, 160, 0),  # stop sign
        (0, 255, 0),  # speed sign
    )

    # New bev observation parameters
    self.use_new_bev_obs = False  # Whether to use bev_observation.py instead of chauffeurnet.py
    self.route_width = 16  # Width of the rendered route in pixel.
    self.red_light_thickness = 3  # Width of the red light line
    self.use_extra_control_inputs = False  # Whether to use extra control inputs such as integral of past steering.
    # Rough avg steering angle in degree that the wheel can be set to
    # The steering angle for individual wheels is  +- 70° and +-48° for the other wheel respectively
    self.max_avg_steer_angle = 60.0
    self.condition_outside_junction = True  # Whether to render the route outside junctions.
    self.use_target_point = False  # Whether to input a target point in the measurements.
    self.use_value_measurements = True  # Whether to use value measurements (otherwise all are set to 0)
    self.num_value_measurements = 3  # Number of measurements exclusive to the value head.
    self.render_speed_lines = False  # Whether to render the speed lines for moving objects.
    self.use_positional_encoding = False  # Whether to add positional encoding to the image
    self.render_yellow_time = False  # Whether to indicate the remaining time to red in yellow light rendering
    self.render_shoulder = True  # Whether to render shoulder lanes as roads.
    self.use_shoulder_channel = False  # Whether to use an extra channel for shoulder lanes

    self.scale_bbox = True  # Whether to scale up the bounding boxes extends 1.0 for vehicles, 2.0 for ped. 0.8 max
    self.scale_factor_vehicle = 1.0
    self.scale_factor_walker = 2.0
    self.min_ext_bounding_box = 0.8
    self.scale_mask_col = 1.0  # Scaling factor for ego vehicle bounding box.
    self.map_folder = 'maps_low_res'  # Map folder for the preprocessed map data
    self.max_speed_actor = 33.33  # In m/s maximum speed we expect from other actors. = 120 km/h
    self.min_speed_actor = -2.67  # In m/s minimum speed we expect from other actors. = -10 km/h

    # Extent of the ego vehicles bounding box
    self.ego_extent_x = 2.44619083404541
    self.ego_extent_y = 0.9183566570281982
    self.ego_extent_z = 0.7451388239860535

    # Roach reward hyperparameters. rr stands for roach reward
    self.reward_type = 'roach'  # Reward function to be used during training. Options: roach, simple_reward
    self.use_exploration_suggest = True  # Whether to use the exploration loss from roach.
    self.rr_maximum_speed = 6.0  # Maximum speed in m/s encouraged by the roach reward function.
    self.vehicle_distance_threshold = 15  # Distance in meters within which vehicles are considered for the reward.
    self.max_vehicle_detection_number = 10  # Maximum number of vehicles considered for the roach reward.
    self.rr_vehicle_proximity_threshold = 9.5  # Threshold within which vehicles are considered hazard in the reward.
    # Distance in meters within which pedestrians are considered for the reward.
    self.pedestrian_distance_threshold = 15
    self.max_pedestrian_detection_number = 10  # Maximum number of pedestrians considered for the roach reward.
    # Threshold within which pedestrians are considered hazard in the reward.
    self.rr_pedestrian_proximity_threshold = 9.5
    self.rr_tl_offset = -0.8 * self.ego_extent_x  # Probably offset to be kept to the entrance of the intersection.
    self.rr_tl_dist_threshold = 18.0  # Distance at which traffic lights are considered for the speed reward.
    # Meters. If the agent is father away from the centerline (laterally) it counts as route deviation in the reward
    self.min_thresh_lat_dist = 3.5
    self.eval_time = 1200.0  # Seconds. After this time a timeout is triggered in the reward which counts as truncation.
    # Number of frames before the end of the episode where the exploration loss is applied.
    self.n_step_exploration = 100
    # If true rr_maximum_speed will be overwritten to the current speed limit affecting the ego vehicle.
    self.use_speed_limit_as_max_speed = False

    # Simple reward hyperparameters
    self.consider_tl = True  # If set to false traffic light infractions are turned off. Used in simple reward
    self.terminal_reward = 0.0  # Reward at the end of the episode
    self.terminal_hint = 10.0  # Reward at the end of the episode when colliding, the number will be subtracted.
    self.normalize_rewards = False  # Whether to use gymnasiums reward normalization.
    self.speeding_infraction = False  # Whether to terminate the route if the agent drives too fast.
    self.use_comfort_infraction = False  # Whether to apply a soft penalty if comfort limits are exceeded
    # These values are tuned for the nuPlan dataset
    self.max_abs_lon_jerk = 30.0  # m/s^3 Comfort limit for longitudinal jerk, nuPlan 4.13
    self.max_abs_mag_jerk = 30.0  # m/s^3 Comfort limit for jerk magnitude, nuPlan 8.37
    self.min_lon_accel = -20.0  # m/s^2 Comfort limit for longitudinal acceleration, nuPlan -4.05
    self.max_lon_accel = 10.0  # m/s^2 Comfort limit for longitudinal acceleration nuPlan 2.40
    self.max_abs_lat_accel = 9.0  # m/s^2 Comfort limit for lateral acceleration, nuPlan 4.89
    self.max_abs_yaw_rate = 1.0  # rad/s Comfort limit for angular velocity, nuPlan 0.95
    self.max_abs_yaw_accel = 3.0  # rad/s^2 Comfort limit for angular yaw acceleration, nuPlan 1.93
    self.comfort_penalty_ticks = 500  # Number of simulator steps that comfort penalties are applied for
    self.comfort_penalty_factor = 0.5  # Max comfort penalty if all comfort metrics are violated.
    self.use_vehicle_close_penalty = False  # Whether to use a penalty for being too close to the front vehicle.
    # Whether to give a penalty depending on vehicle speed when crashing or running red light
    self.use_termination_hint = False
    self.ego_forecast_time = 1.0  # Number of seconds that the ego agent is forecasted.
    self.ego_forecast_min_speed = 2.5  # In m/s. Minimum speed in the ego forecast.
    self.use_perc_progress = False  # Whether to multiply RC reward by percentage away from lane center.
    self.lane_distance_violation_threshold = 0.0  # Grace distance in m at which no lane perc penalty is applied
    self.lane_dist_penalty_softener = 1.0  # If smaller than 1 reduces lane distance penalty.
    self.use_min_speed_infraction = False  # Whether to penalize the agent for driving slower than other agents on avg.
    self.use_leave_route_done = True  # Whether to terminate the route when leaving the precomputed path.
    self.use_outside_route_lanes = False  # Whether to terminate the route when invading opposing lanes or sidewalks.
    self.use_max_change_penalty = False  # Whether to apply a soft penalty when the action changes too fast.
    self.max_change = 0.25  # Maximum change in action allowed compared to last frame before a penalty is applied
    self.penalize_yellow_light = True  # Whether to penalize running a yellow light.
    self.use_off_road_term = False  # Whether to terminate when he agent drives off the drivable area
    self.off_road_term_perc = 0.5  # Percentage of agent overlap with off-road, that triggers the termination
    # Whether to use a different stop sign detector that prevents the policy from cheating by changing lanes.
    self.use_new_stop_sign_detector = False
    self.use_ttc = False  # Whether to use TTC in the reward.
    self.ttc_resolution = 2  # Interval of frame_rate time steps at which TTC is evaluated
    self.ttc_penalty_ticks = 500  # Number of simulator steps that a TTC penalty is applied for
    # m/s. Number adopted from nuPlan.
    # If the agent exceeds speed limit within this threshold he will get a penalty between [0,1] linearly
    self.max_overspeed_value_threshold = 2.23
    # Whether to only use RC als reward source in simple reward, else adds TTC, comfort and speed like in nuPlan
    self.use_single_reward = True
    self.use_rl_termination_hint = False  # Whether to include red light infraction for termination hints
    self.use_survival_reward = False  # Whether to add a constant reward every frame
    self.survival_reward_magnitude = 0.0001  # How large the survival reward is.

    # How often an action is repeated.
    self.action_repeat = 1

    # Action and observation space
    self.obs_num_measurements = 8  # Number of scalar measurements in observation.
    self.obs_num_channels = 15  # Number of channels in the bev observation.

    ####### Distribution parameters ############
    self.distribution = 'beta'  # Distribution used for the action space. Options beta, normal, beta_uni_mix
    # Minimum value for a, b of the beta distribution that the model can predict. Gets added to the softplus output.
    self.beta_min_a_b_value = 1.0

    self.normal_dist_init = ((0, -2), (0, -2))  # Initial bias parameters of the normal distribution
    self.normal_dist_action_dep_std = True  # Whether the std of the normal distribution is dependent of the input

    self.uniform_percentage_z = 0.03  # Mixing percentage of uniform distribution in beta_uni_mix

    # We have 2 actions, corresponding to left right steering and negative to positive acceleration.
    self.action_space_dim = 2
    self.action_space_min = -1.0  # Minimum value of the action space
    self.action_space_max = 1.0  # Maximum value of the action space
    # Number of frames at the beginning before learning starts, return brake
    self.start_delay_frames = int(2.0 / self.time_interval + 0.5)

    # PPO training hyperparameters
    self.exp_name = 'PPO_000'  # the name of this experiment
    self.gym_id = 'CARLAEnv-v0'  # the id of the gym environment
    self.learning_rate = 1.0e-5  # the learning rate of the optimizer
    self.seed = 1  # seed of the experiment
    self.total_timesteps = 10000000  # total time steps of the experiments
    self.torch_deterministic = True  # if toggled, `torch.backends.cudnn.deterministic=False`
    self.cuda = True  # if toggled, cuda will be enabled by default
    self.track = False  #if toggled, this experiment will be tracked with Weights and Biases
    self.wandb_project_name = 'ppo-roach'  # the wandb project name
    self.wandb_entity = None  # the entity (team) of wandb project
    self.capture_video = False  # whether to capture videos of the agent performances (check out `videos` folder)
    self.num_envs = 1  # the number of parallel game environments
    self.lr_schedule = 'kl'  # Which lr schedule to use. Options: (linear, kl, none, step, cosine, cosine_restart)
    self.gae = True  # Use GAE for advantage computation
    self.gamma = 0.99  # the discount factor gamma
    self.gae_lambda = 0.95  # the lambda for the general advantage estimation
    self.update_epochs = 4  # the K epochs to update the policy
    self.norm_adv = False  # Toggles advantages normalization
    self.clip_coef = 0.1  # the surrogate clipping coefficient
    self.clip_vloss = False  # Toggles whether to use a clipped loss for the value function, as per the paper.
    self.ent_coef = 0.01  # coefficient of the entropy
    self.vf_coef = 0.5  # coefficient of the value function
    self.max_grad_norm = 0.5  # the maximum norm for the gradient clipping
    self.target_kl = 0.015  # the target KL divergence threshold
    self.visualize = False  # if toggled, Game will render on screen
    self.logdir = ''  # The directory to log the data into.
    self.load_file = None  # model weights for initialization
    # Ports of the carla_gym wrapper to connect to. It requires to submit a port for every envs ports == --num_envs
    self.ports = (5555,)
    self.gpu_ids = (0,)  # Which GPUs to train on. Index 0 indicates GPU for rank 0 etc.
    self.compile_model = False  # Whether to use torch compile on the model.
    self.total_batch_size = 512  # The total amount of data collected at every step across all environments
    self.total_minibatch_size = 256  # The total minibatch sized used for training (across all environments)
    self.expl_coef = 0.05  # Weight / coefficient of the exploration loss
    self.lr_schedule_step = 8  # Number of time the KL divergence can be triggered before the lr reduces.
    self.current_learning_rate = self.learning_rate  # Learning rate at the latest iteration.
    self.kl_early_stop = 0  # Counter that reduces lr once it reaches lr_schedule_step
    self.adam_eps = 1e-5  # Adam optimizer parameter parameter. Standard PPO value is 1e-5
    # Did not observe a significant speedup with these so we turn them off for better numerical precision.
    self.allow_tf32 = False  # Whether to use tf32 format, which has better speed but lower numeric precision.
    self.benchmark = False  # Whether to use cudnn benchmarking
    self.matmul_precision = 'highest'  # Options highest float32, high tf32, medium bfloat16
    # Whether to collect data on cpu. This can be a bit faster, since it avoid CPU GPU ping pong,
    # at the cost of running the model on the CPU during data collection.
    self.cpu_collect = False
    # Robust policy optimization https://arxiv.org/abs/2212.07536
    self.use_rpo = False
    self.rpo_alpha = 0.5  # Size of the uniform random value that gets added to a, b
    self.use_green_wave = False  # If true in some routes all TL that the agent encounters are set to green.
    self.green_wave_prob = 0.05  # Probability of a route using green wave (if use_green_wave=True)
    # You should pick tiny networks for efficiency e.g. convnext_atto.d2_in1k,
    #  tinynet_e.in1k, vit_small_patch32_224
    self.image_encoder = 'roach'  # Which image cnn encoder to use. Either roach, roach_ln, or timm model name
    self.use_layer_norm = False  # Whether to use LayerNorm before ReLU in MLPs.
    # Applicable if use_layer_norm=True, whether to also apply layernorm to the policy head.
    # Can be useful to remove to allow the policy to predict large values (for a, b of Beta).
    self.use_layer_norm_policy_head = True
    self.features_dim = 256  # Dimension of features produced by the state encoder
    self.use_lstm = False  # Whether to use an LSTM after the feature encoder.
    self.num_lstm_layers = 1  # How many LSTM layers to use.


    self.render_green_tl = True  # Whether to render green traffic lights into the observation.
    self.lr_schedule_step_factor = 0.1  # Multiplier when doing a step decrease in learning rate
    self.lr_schedule_step_perc = (0.5, 0.75)  # Percentage of training run after which the lr is decayed
    self.weight_decay = 0.0  # Weight decay applied to optimizer. AdamW is used when > 0.0
    self.beta_1 = 0.9  # Beta 1 parameter of adam
    self.beta_2 = 0.999  # Beta 2 parameter of adam
    self.lr_schedule_cosine_restarts = (0.0, 0.25, 0.50, 0.75, 1.0)  # Percentage of training to do a restart
    # https://arxiv.org/abs/1911.00357
    self.use_dd_ppo_preempt = False  # Whether to use the dd-ppo preemption technique to early stop stragglers
    self.dd_ppo_preempt_threshold = 0.6  # Percentage of nodes that need to be finished before the rest is stopped.
    self.dd_ppo_min_perc = 0.25  # Minimum percentage of data points that need to be collected before preemption.
    self.num_envs_per_proc = 1  # Number of environments to use per process.
    # Percentage of training at which the model is evaluated
    self.eval_intervals = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    self.current_eval_interval_idx = 0  # Helper variable to remember which model to save next.
    self.use_temperature = False  # Whether the output distribution parameters are divided by a learned temperature
    self.min_temperature = 0.1  # Whether the output distribution parameters are divided by a learned temperature

    # Whether to use the histogram loss gauss to train the value head via classification (instead of regression + L2)
    self.use_hl_gauss_value_loss = False
    self.hl_gauss_std = 0.75  # Standard deviation use for the gaussian histogram loss
    self.hl_gauss_vmin = -10.0  # Min value of the histogram in HL_Gauss. Tune to be in return range
    self.hl_gauss_vmax = 30.0  # Max value of the histogram in HL_Gauss. Tune to be in return range
    self.hl_gauss_bucket_size = 1.0  # Size of each bucket in the HL_Gauss histogram.
    self.hl_gauss_num_classes = int((self.hl_gauss_vmax - self.hl_gauss_vmin) / self.hl_gauss_bucket_size) + 1

    self.global_step = 0  # Current iteration of the training
    self.max_training_score = -np.inf  # Highest training score achieved so far
    self.best_iteration = 0  # Iteration of the best model
    self.latest_iteration = 0  # Iteration of the latest model

  def initialize(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
