"""
Connects with a carla gym wrapper.
"""
# Python imports
import os
# Set CUBLAS workspace config before torch import to enable deterministic cuBLAS operations
# Required when torch.use_deterministic_algorithms(True) is used with CUDA >= 10.2
# See: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from collections import deque
import math
import pathlib
import subprocess

# Carla leaderboard imports
from leaderboard_codes import autonomous_agent2 as autonomous_agent
from leaderboard_codes.carla_data_provider import CarlaDataProvider

# Pip imports
import numpy as np
import carla
import torch
import torch.nn.functional as F
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import cv2
from gymnasium import spaces
from pytictoc import TicToc
import zmq

# Code imports
from rl_config import GlobalConfig
import rl_utils as rl_u
from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.bev_observation import ObsManager as ObsManager2
from birds_eye_view.run_stop_sign import RunStopSign
from nav_planner import RoutePlanner
from model import PPOPolicy
from reward.roach_reward import RoachReward
from reward.simple_reward import SimpleReward

jsonpickle_numpy.register_handlers()

# Make network deterministic.
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'EvalAgent'


class EvalAgent(autonomous_agent.AutonomousAgent):
  """
    Main class that runs the agents with the run_step function
    """

  def setup(self, path_to_conf_file, route_index=None):  # pylint: disable=locally-disabled, unused-argument
    """Sets up the agent. route_index is for logging purposes"""
    self.step = -1
    self.track = autonomous_agent.Track.MAP
    self.config_path = path_to_conf_file
    self.config = GlobalConfig()

    with open(os.path.join(path_to_conf_file, 'config.json'), 'rt', encoding='utf-8') as f:
      json_config = f.read()

    # 4 ms, might need to move outside the agent.
    loaded_config = jsonpickle.decode(json_config)

    # Overwrite all properties that were set in the saved config.
    self.config.__dict__.update(loaded_config.__dict__)

    self.initialized = False
    self.list_traffic_lights = []

    # Environment variables
    self.save_path = os.environ.get('SAVE_PATH', None)
    self.config.debug = int(os.environ.get('DEBUG_ENV_AGENT', 0)) == 1
    self.sample_type = os.environ.get('SAMPLE_TYPE', 'mean')  # Options: roach, mean, sample
    self.record_infractions = int(os.environ.get('RECORD', 0)) == 1
    self.cpp = int(os.environ.get('CPP', 0)) == 1  # Whether to evaluate a model trained with c++
    self.port = int(os.environ.get('CPP_PORT', 5555))  # Port over which to do communication
    self.upscale_factor = int(os.environ.get('UPSCALE_FACTOR', 1))  # Increases resolution if visualizations
    self.save_png = int(os.environ.get('SAVE_PNG', 0)) == 1  # Save renderings also as individual PNG
    # We train at 10 Hz, this flag can be used to run the policy at 20 Hz during inference.
    # Only sensible for policies whose inputs are time step independent.
    self.high_freq_inference = int(os.environ.get('HIGH_FREQ_INFERENCE', 0))
    print('Save_path: ', self.save_path)
    print('DEBUG_ENV_AGENT: ', self.config.debug)
    print('SAMPLE_TYPE: ', self.sample_type)
    print('RECORD: ', self.record_infractions)
    print('CPP: ', self.cpp)
    print('HIGH_FREQ_INFERENCE: ', self.high_freq_inference)

    # CPU seems faster
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self.max_speed = 0.0  # Maximum speed driven during the evaluation in m/s

    self.agents = []
    self.model_count = 0  # Counts how many models are in our ensemble
    self.observation_space = spaces.Dict({
        'bev_semantics':
            spaces.Box(0,
                       255,
                       shape=(self.config.obs_num_channels, self.config.bev_semantics_height,
                              self.config.bev_semantics_width),
                       dtype=np.uint8),
        'measurements':
            spaces.Box(-math.inf, math.inf, shape=(self.config.obs_num_measurements,), dtype=np.float32)
    })
    self.action_space = spaces.Box(self.config.action_space_min,
                                   self.config.action_space_max,
                                   shape=(self.config.action_space_dim,),
                                   dtype=np.float32)
    for file in os.listdir(path_to_conf_file):
      if file.startswith('model') and file.endswith('.pth'):
        self.model_count += 1
        if not self.cpp:
          print(os.path.join(path_to_conf_file, file))
          agent = PPOPolicy(self.observation_space, self.action_space, config=self.config).to(self.device)
          if self.config.compile_model:
            agent = torch.compile(agent)

          state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location=self.device)
          agent.load_state_dict(state_dict, strict=True)
          agent.to(self.device)
          agent.eval()
          self.agents.append(agent)

    if self.cpp:
      current_folder = pathlib.Path(__file__).parent.resolve()
      comm_folder = os.path.join(current_folder, 'comm_files')

      ppo_cpp_install_path = os.environ.get('PPO_CPP_INSTALL_PATH')
      path_to_comm = str(comm_folder)
      path_to_singularity = os.environ.get('PATH_TO_SINGULARITY')
      torch_kernel_cache = os.environ.get('PYTORCH_KERNEL_CACHE_PATH', '~/.cache/torch')

      print('Path to conf file:', path_to_conf_file, flush=True)
      print(
          f'singularity exec --nv --env LD_LIBRARY_PATH={ppo_cpp_install_path}:$LD_LIBRARY_PATH '
          f'--env PYTORCH_KERNEL_CACHE_PATH={torch_kernel_cache} --bind {path_to_conf_file}:{path_to_conf_file},'
          f'{ppo_cpp_install_path}:{ppo_cpp_install_path},{path_to_comm}:{path_to_comm},{torch_kernel_cache}'
          f':{torch_kernel_cache} {path_to_singularity} {ppo_cpp_install_path}/ppo_carla_inference '
          f'--path_to_conf_file {path_to_conf_file} --ipc_path {path_to_comm} --port {self.port}',
          flush=True)
      # Starts the C++ process that runs the model in a singularity container.
      _ = subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
          f'singularity exec --nv --env LD_LIBRARY_PATH={ppo_cpp_install_path}:$LD_LIBRARY_PATH '
          f'--env PYTORCH_KERNEL_CACHE_PATH={torch_kernel_cache} --bind {path_to_conf_file}:{path_to_conf_file},'
          f'{ppo_cpp_install_path}:{ppo_cpp_install_path},{path_to_comm}:{path_to_comm},{torch_kernel_cache}'
          f':{torch_kernel_cache} {path_to_singularity} {ppo_cpp_install_path}/ppo_carla_inference '
          f'--path_to_conf_file {path_to_conf_file} --ipc_path {path_to_comm} --port {self.port}',
          shell=True)
      self.context = zmq.Context()
      self.socket = self.context.socket(zmq.PAIR)
      pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
      communication_file = os.path.join(comm_folder, str(self.port))
      # Connect to python process receiving up to date config file.
      self.socket.connect(f'ipc://{communication_file}.lock')
      message = self.socket.recv_string()
      print(message)
      self.socket.send_string(self.sample_type)

      # For visualization
      self.agents.append(
          PPOPolicy(self.observation_space, self.action_space, config=self.config).to(self.device).eval())

    if self.config.debug and self.save_path is not None:
      self.route_index = route_index
      self.visu_image_buffer = deque(maxlen=100000)
      self.collected_rewards = []

    if self.save_path is not None and self.record_infractions:
      self.config.penalize_yellow_light = False
      self.config.eval_time = 2000  # Don't want to log timeout infractions.
      self.route_index = route_index
      self.infraction_buffer = deque(maxlen=int(5.0 * self.config.frame_rate))
      self.infraction_counter = 0  # Number of logged infractions during this route.

    if self.high_freq_inference:
      self.total_action_repeat = int(self.config.action_repeat)
    else:
      self.total_action_repeat = int(self.config.action_repeat *
                                     (self.config.original_frame_rate // self.config.frame_rate))

  def sensors(self):
    sensors = []

    return sensors

  def agent_init(self, vehicle):
    self.vehicle = vehicle
    self.world = self.vehicle.get_world()
    self.world_map = self.world.get_map()
    self.stop_sign_criteria = RunStopSign(self.world, self.world_map)
    self.vehicles_all = []
    self.walkers_all = []
    if self.config.use_new_bev_obs:
      self.bev_semantics_manager = ObsManager2(self.config)
      self.bev_semantics_manager.attach_ego_vehicle(self.vehicle, self.stop_sign_criteria, self.world_map,
                                                    self.dense_global_plan_world_coord)

    else:
      self.bev_semantics_manager = ObsManager(self.config)
      self.bev_semantics_manager.attach_ego_vehicle(self.vehicle, self.stop_sign_criteria, self.world_map)

    # Preprocess traffic lights
    all_actors = self.world.get_actors()
    for actor in all_actors:
      if 'traffic_light' in actor.type_id:
        center, waypoints = rl_u.get_traffic_light_waypoints(actor, self.world_map)
        self.list_traffic_lights.append((actor, center, waypoints))

    self.route_planner = RoutePlanner()
    self.route_planner.set_route(self.dense_global_plan_world_coord)
    self.total_route_len = len(self.dense_global_plan_world_coord)

    if self.config.reward_type == 'roach':
      self.reward_handler = RoachReward(self.vehicle, self.world_map, self.world, self.config)
    elif self.config.reward_type == 'simple_reward':
      self.reward_handler = SimpleReward(self.vehicle, self.world_map, self.world, self.config,
                                         self.dense_global_plan_world_coord)

    if self.config.use_extra_control_inputs:
      self.last_wheel_angle = 0.0
      self.past_wheel_errors = deque([0.0 for _ in range(int(1.0 * self.config.frame_rate))],
                                     maxlen=int(1.0 * self.config.frame_rate))

    if self.config.use_hl_gauss_value_loss:
      self.hl_gauss_bins = rl_u.hl_gaus_bins(self.config.hl_gauss_vmin, self.config.hl_gauss_vmax,
                                             self.config.hl_gauss_bucket_size, self.device)

    self.last_lstm_states = []
    for _ in range(self.model_count):
      self.last_lstm_states.append((
          torch.zeros(self.config.num_lstm_layers, 1, self.config.features_dim, device=self.device),
          torch.zeros(self.config.num_lstm_layers, 1, self.config.features_dim, device=self.device),
      ))
    self.done = torch.zeros(1, device=self.device)

    self.measured_times = []

    if self.save_path is not None and self.record_infractions:
      self.last_infraction_location = self.vehicle.get_location()

    self.initialized = True

  def preprocess_observation(self, waypoint_route, timestamp):
    speed = self.vehicle.get_velocity().length()

    if speed > self.max_speed:
      self.max_speed = speed

    self.stop_sign_criteria.tick(self.vehicle)
    actors = self.world.get_actors()
    self.vehicles_all = actors.filter('*vehicle*')
    self.walkers_all = actors.filter('*walker*')
    self.static_all = actors.filter('*static*')
    debug = (self.config.debug or self.record_infractions) and self.save_path is not None
    bev_semantics = self.bev_semantics_manager.get_observation(waypoint_route,
                                                               self.vehicles_all,
                                                               self.walkers_all,
                                                               self.static_all,
                                                               debug=debug)
    observations = {'bev_semantics': bev_semantics['bev_semantic_classes']}

    if debug:
      observations['rendered'] = bev_semantics['rendered']

    last_control = self.vehicle.get_control()
    velocity = self.vehicle.get_velocity()
    transform = self.vehicle.get_transform()
    forward_vec = transform.get_forward_vector()

    np_vel = np.array([velocity.x, velocity.y, velocity.z])
    np_fvec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
    forward_speed = np.dot(np_vel, np_fvec)

    np_vel_2d = np.array([velocity.x, velocity.y])
    velocity_ego_frame = rl_u.inverse_conversion_2d(np_vel_2d, np.zeros(2), np.deg2rad(transform.rotation.yaw))

    # acceleration = self.vehicle.get_acceleration()
    # np_acceleration_2d = np.array([acceleration.x, acceleration.y])
    # acc_ego_frame = rl_u.inverse_conversion_2d(np_acceleration_2d, np.zeros(2), np.deg2rad(transform.rotation.yaw))

    speed_limit = self.vehicle.get_speed_limit()
    if isinstance(speed_limit, float):
      # Speed limit is in km/h we compute with m/s, so we convert it by / 3.6
      maximum_speed = speed_limit / 3.6
    else:
      #  Car can have no speed limit right after spawning
      maximum_speed = self.config.rr_maximum_speed

    measurements = [
        last_control.steer, last_control.throttle, last_control.brake,
        float(last_control.gear),
        float(velocity_ego_frame[0]),
        float(velocity_ego_frame[1]),
        float(forward_speed), maximum_speed
    ]

    if self.config.use_extra_control_inputs:
      left_wheel = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
      right_wheel = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
      avg_wheel = 0.5 * (left_wheel + right_wheel)  # They can be quite different, we take the avg to simplify.
      avg_wheel /= self.config.max_avg_steer_angle  # Normalize from range [-60, 60] to [-1, 1]

      measurements.append(avg_wheel)

      last_error = last_control.steer - self.last_wheel_angle

      self.past_wheel_errors.append(last_error)
      # I am omitting the time step because it is constant, normalizes the input automatically in [-1, 1]
      error_derivative = self.past_wheel_errors[-1] - self.past_wheel_errors[-2]
      error_integral = sum(self.past_wheel_errors) / len(self.past_wheel_errors)

      # These inputs should allow the model to learn something like a PID controller for steering.
      measurements.append(last_error)
      measurements.append(error_derivative)
      measurements.append(error_integral)

      self.last_wheel_angle = avg_wheel

    if self.config.use_target_point:
      measurements.append(bev_semantics['target_point'][0])
      measurements.append(bev_semantics['target_point'][1])

    observations['measurements'] = np.array(measurements, dtype=np.float32)

    # Add remaining time till timeout. remaining time till blocked, remaining route to help return prediction
    remaining_time = (self.config.eval_time - timestamp) / self.config.eval_time
    time_till_blocked = self.reward_handler.block_detector.time_till_blocked
    perc_route_left = float(len(waypoint_route)) / 100.0  # 100.0 is just some constant for normalization
    if self.config.use_ttc:
      remaining_ttc_penalty_ticks = self.reward_handler.remaining_ttc_penalty_ticks / self.config.ttc_penalty_ticks
    if self.config.use_comfort_infraction:
      remaining_comfort_penalty_ticks = (self.reward_handler.remaining_comfort_penalty_ticks /
                                         self.config.comfort_penalty_ticks)

    if not self.config.use_value_measurements:
      remaining_time = 0.0
      time_till_blocked = 0.0
      perc_route_left = 0.0
      if self.config.use_ttc:
        remaining_ttc_penalty_ticks = 0.0
      if self.config.use_comfort_infraction:
        remaining_comfort_penalty_ticks = np.zeros(6)

    value_measurements = [remaining_time, time_till_blocked, perc_route_left]
    if self.config.use_ttc:
      value_measurements.append(remaining_ttc_penalty_ticks)
    if self.config.use_comfort_infraction:
      value_measurements.extend(remaining_comfort_penalty_ticks)

    assert self.config.num_value_measurements == len(value_measurements)

    observations['value_measurements'] = np.array(value_measurements, dtype=np.float32)

    collision_with_pedestrian = bev_semantics['collision_px']
    perc_off_road = bev_semantics['percentage_off_road']

    return observations, collision_with_pedestrian, perc_off_road

  def get_waypoint_route(self):
    ego_vehicle_transform = self.vehicle.get_transform()
    pos = ego_vehicle_transform.location
    pos = np.array([pos.x, pos.y])
    waypoint_route = self.route_planner.run_step(pos)

    return waypoint_route

  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, input_data, timestamp, vehicle, sensors=None):  # pylint: disable=locally-disabled, unused-argument
    self.step += 1

    t = TicToc()
    t.tic()

    self.last_timestamp = timestamp
    if not self.initialized:
      self.agent_init(vehicle)
      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      self.last_control = control
      return control

    # Evaluate without vehicles.
    if int(os.environ.get('NO_CARS', 0)) == 1:
      all_vehicles = CarlaDataProvider.get_all_actors().filter('vehicle*')
      for vehicle in all_vehicles:
        if vehicle.id != self.vehicle.id:
          vehicle.destroy()

    if self.step % self.total_action_repeat != 0:
      return self.last_control

    waypoint_route = self.get_waypoint_route()
    obs, collision_with_pedestrian, perc_off_road = self.preprocess_observation(waypoint_route, timestamp)

    if self.save_path is not None and (self.config.debug or self.record_infractions):
      reward, termination, _, info = self.reward_handler.get(timestamp, waypoint_route, collision_with_pedestrian,
                                                             self.vehicles_all, self.walkers_all, self.static_all,
                                                             perc_off_road)
      if self.config.debug:
        self.collected_rewards.append(reward)

    actions = []
    if self.cpp:
      self.socket.send_string('')  # Means keep connection alive
      self.socket.send_multipart((obs['bev_semantics'], obs['measurements'], obs['value_measurements']), copy=False)
      message = self.socket.recv_multipart(copy=False)
      action = np.frombuffer(message[0], dtype=np.float32)
      if self.save_path is not None and (self.config.debug or self.record_infractions):
        value = torch.from_numpy(np.frombuffer(message[1], dtype=np.float32)).to(self.device)
        alpha = torch.from_numpy(np.frombuffer(message[2], dtype=np.float32)).to(self.device)
        beta = torch.from_numpy(np.frombuffer(message[3], dtype=np.float32)).to(self.device)
        if self.config.distribution == 'beta':
          distribution = torch.distributions.Beta(alpha, beta)
        else:
          raise ValueError('Distribution selected that is not implemented. Options: beta')

      pred_sem = None  # World model loss not implemented in C++
      pred_measure = None
      self.action = action
    else:
      obs_tensor = {
          'bev_semantics':
              torch.Tensor(obs['bev_semantics'][np.newaxis, ...]).to(self.device, dtype=torch.float32),
          'measurements':
              torch.Tensor(obs['measurements'][np.newaxis, ...]).to(self.device, dtype=torch.float32),
          'value_measurements':
              torch.Tensor(obs['value_measurements'][np.newaxis, ...]).to(self.device, dtype=torch.float32)
      }

      for i in range(self.model_count):
        action, _, _, value, _, _, _, distribution, pred_sem, pred_measure, self.last_lstm_states[i] = \
          self.agents[i].forward(obs_tensor, sample_type=self.sample_type, lstm_state=self.last_lstm_states[i],
                                 done=self.done)
        if self.config.use_hl_gauss_value_loss:
          value_pdf = F.softmax(value, dim=1)
          value = torch.sum(value_pdf * self.hl_gauss_bins.unsqueeze(0), dim=1)
        actions.append(action)

      self.action = torch.stack(actions, dim=0).mean(dim=0)[0].cpu().numpy()

    control = self.convert_action_to_control(self.action)
    self.last_control = control

    if self.save_path is not None and (self.config.debug or self.record_infractions):
      visualization = self.agents[-1].visualize_model(distribution, obs['rendered'], obs['measurements'], control,
                                                      value, obs['value_measurements'], pred_sem, pred_measure,
                                                      self.upscale_factor)
    # Render action distribution and model speed into visualization
    if self.config.debug and self.save_path is not None:
      self.visu_image_buffer.append(visualization)
    if self.record_infractions and self.save_path is not None:
      self.infraction_buffer.append(visualization)

      if termination and (self.last_infraction_location.distance(self.vehicle.get_location()) > 10.0):
        self.infraction_counter += 1
        self.save_infraction_clip(info['infraction_type'])
        self.last_infraction_location = self.vehicle.get_location()

    self.measured_times.append(t.tocvalue())
    return control

  def convert_action_to_control(self, action):
    # Convert acceleration to brake / throttle. Acc in [-1,1]. Negative acceleration -> brake
    if action[1] > 0.0:
      throttle = action[1]
      brake = 0.0
    else:
      throttle = 0.0
      brake = -action[1]
    control = carla.VehicleControl(steer=float(action[0]), throttle=float(throttle), brake=float(brake))
    return control

  def save_infraction_clip(self, infraction_type):
    if len(self.infraction_buffer) <= 0:
      return

    video_save_path = os.path.join(
        self.save_path,
        f'{self.config.exp_name}_{self.route_index}_{infraction_type}_{self.infraction_counter:02d}.avi')
    height, width, _ = self.infraction_buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # VP90 slower but compresses 2x better
    video = cv2.VideoWriter(video_save_path, fourcc, int(self.config.frame_rate), (width, height))
    for image in self.infraction_buffer:
      video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    """
    Gets called after a route finished.
    The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
    Also writes logging files to disk.
    """
    waypoint_route = self.get_waypoint_route()
    _, collision_with_pedestrian, perc_off_road = self.preprocess_observation(waypoint_route, self.last_timestamp)
    _, termination, _, info = self.reward_handler.get(self.last_timestamp, waypoint_route, collision_with_pedestrian,
                                                      self.vehicles_all, self.walkers_all, self.static_all,
                                                      perc_off_road)

    if self.record_infractions and self.save_path is not None:
      if termination and (self.last_infraction_location.distance(self.vehicle.get_location()) > 10.0):
        self.infraction_counter += 1
        self.save_infraction_clip(info['infraction_type'])
        self.last_infraction_location = self.vehicle.get_location()

    if len(self.measured_times) > 0:
      print('Avg. run_step:', sum(self.measured_times) / len(self.measured_times))

    # For tuning comfort values.
    # from matplotlib import pyplot as plt
    # plt.hist(self.reward_handler.comfort_histogram['acc_lon'])
    # plt.axvline(x=self.config.max_lon_accel, color='r', linestyle='--', label='max')
    # plt.axvline(x=self.config.min_lon_accel, color='b', linestyle='--', label='min')
    # plt.ylabel('acc_lon')
    # plt.show()
    # plt.clf()
    #
    # plt.hist(self.reward_handler.comfort_histogram['acc_lat'])
    # plt.axvline(x=self.config.max_abs_lat_accel, color='r', linestyle='--', label='max')
    # plt.ylabel('acc_lat')
    # plt.show()
    # plt.clf()
    #
    #
    # plt.hist(self.reward_handler.comfort_histogram['jerk'])
    # plt.axvline(x=self.config.max_abs_mag_jerk, color='r', linestyle='--', label='max')
    # plt.ylabel('jerk')
    # plt.show()
    # plt.clf()
    #
    #
    # plt.hist(self.reward_handler.comfort_histogram['jerk_lon'])
    # plt.axvline(x=self.config.max_abs_lon_jerk, color='r', linestyle='--', label='max')
    # plt.ylabel('jerk_lon')
    # plt.show()
    # plt.clf()
    #
    #
    # plt.hist(self.reward_handler.comfort_histogram['yaw_rate'])
    # plt.axvline(x=self.config.max_abs_yaw_rate, color='r', linestyle='--', label='max')
    # plt.ylabel('yaw_rate')
    # plt.show()
    # plt.clf()
    #
    #
    # plt.hist(self.reward_handler.comfort_histogram['yaw_acceleration'])
    # plt.axvline(x=self.config.max_abs_yaw_accel, color='r', linestyle='--', label='max')
    # plt.ylabel('yaw_acceleration')
    # plt.show()
    # plt.clf()

    print(f'Max driving speed: {self.max_speed * 3.6} km/h.')
    del self.measured_times
    self.reward_handler.destroy()
    del self.reward_handler
    del self.vehicles_all
    del self.walkers_all

    if self.config.debug and self.save_path is not None:

      if self.save_png:
        png_folder = pathlib.Path(self.save_path) / (self.config.exp_name + '_' + self.route_index)
        png_folder.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(self.visu_image_buffer):
          save_file = png_folder / f'{idx:04d}.png'
          cv2.imwrite(str(save_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

      video_save_path = os.path.join(self.save_path, f'{self.config.exp_name}_{self.route_index}.avi')
      height, width, _ = self.visu_image_buffer[0].shape
      fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # VP90 slower but compresses 2x better
      video = cv2.VideoWriter(video_save_path, fourcc, int(self.config.frame_rate), (width, height))
      for image in self.visu_image_buffer:
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

      cv2.destroyAllWindows()
      video.release()
      del self.visu_image_buffer
      print('Total rewards:', sum(self.collected_rewards))
      # from matplotlib import pyplot as plt
      # xs = np.arange(0, len(self.collected_rewards))
      # plt.plot(xs, np.array(self.collected_rewards))
      # plt.show()
      del self.collected_rewards

    if self.cpp:
      self.socket.send_string('Shutdown')
      self.socket.close()
      self.context.term()
