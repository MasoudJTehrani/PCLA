import numpy as np
# from config import GlobalConfig

class Config:
    def __init__(self, frame_rate=20):
        #  Time step for the model (20 frames per second).
        self.time_step = 1./frame_rate
        # Kinematic bicycle model parameters tuned from World on Rails.
        # Distance from the rear axle to the front axle of the vehicle.
        self.front_wheel_base = -0.090769015
        # Distance from the rear axle to the center of the rear wheels.
        self.rear_wheel_base = 1.4178275
        # Gain factor for steering angle to wheel angle conversion.
        self.steering_gain = 0.36848336
        # Deceleration rate when braking (m/s^2) of other vehicles.
        self.brake_acceleration = -4.952399
        # Acceleration rate when throttling (m/s^2) of other vehicles.
        self.throttle_acceleration = 0.5633837
        # Tuned parameters for the polynomial equations modeling speed changes
        # Numbers are tuned parameters for the polynomial equations below using
        # a dataset where the car drives on a straight highway, accelerates to
        # and brakes again
        # Coefficients for polynomial equation estimating speed change with throttle input for ego model.
        self.throttle_values = np.array([9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02, -1.05996152e-03, 6.71079346e-04])
        # Coefficients for polynomial equation estimating speed change with brake input for the ego model.
        self.brake_values = np.array([9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09, -4.91381935e-12])
        # Minimum throttle value that has an affect during forecasting the ego vehicle.
        self.throttle_threshold_during_forecasting = 0.3


class KinematicBicycleModel():
    """
    Kinematic bicycle model describing the motion of a car given its state and action.
    Tuned parameters are taken from World on Rails.
    """

    def __init__(self, frame_rate=20):
        self.config = Config(frame_rate)

        self.time_step = self.config.time_step
        self.front_wheel_base = self.config.front_wheel_base
        self.rear_wheel_base = self.config.rear_wheel_base
        self.steering_gain = self.config.steering_gain
        self.brake_acceleration = self.config.brake_acceleration
        self.throttle_acceleration = self.config.throttle_acceleration
        self.throttle_values = self.config.throttle_values
        self.brake_values = self.config.brake_values
        self.throttle_threshold_during_forecasting = self.config.throttle_threshold_during_forecasting

    def forecast_other_vehicles(self, locations, headings, speeds, actions):
        """
        Forecast the future states of other vehicles based on their current states and actions.

        Args:
            locations (numpy.ndarray): Array of (x, y, z) coordinates representing the locations of other vehicles.
            headings (numpy.ndarray): Array of heading angles (in radians) for other vehicles.
            speeds (numpy.ndarray): Array of speeds (in m/s) for other vehicles.
            actions (numpy.ndarray): Array of actions (steer, throttle, brake) for other vehicles.

        Returns:
            tuple: A tuple containing the forecasted locations, headings, and speeds for other vehicles.
        """
        steers, throttles, brakes = actions[:, 0], actions[:, 1], actions[:, 2].astype(np.uint8)
        wheel_angles = self.steering_gain * steers
        slip_angles = np.arctan(self.rear_wheel_base / (self.front_wheel_base + self.rear_wheel_base) * np.tan(wheel_angles))
        
        next_x = locations[:, 0] + speeds * np.cos(headings + slip_angles) * self.time_step
        next_y = locations[:, 1] + speeds * np.sin(headings + slip_angles) * self.time_step
        next_headings = headings + speeds / self.rear_wheel_base * np.sin(slip_angles) * self.time_step

        next_speeds = speeds + self.time_step * np.where(brakes, self.brake_acceleration, throttles * self.throttle_acceleration)
        next_speeds = np.maximum(0.0, next_speeds)

        next_locations = np.column_stack([next_x, next_y, locations[:, 2]])

        return next_locations, next_headings, next_speeds

    def forecast_ego_vehicle(self, location, heading, speed, action):
        """
        Forecast the future state of the ego vehicle based on its current state and action.

        Args:
            location (numpy.ndarray): Array of (x, y, z) coordinates representing the location of the ego vehicle.
            heading (float): Current heading angle (in radians) of the ego vehicle.
            speed (float): Current speed (in m/s) of the ego vehicle.
            action (numpy.ndarray): Action (steer, throttle, brake) for the ego vehicle.

        Returns:
            tuple: A tuple containing the forecasted location, heading, and speed for the ego vehicle.
        """
        steer, throttle, brake = action
        wheel_angle = self.steering_gain * steer
        slip_angle = np.arctan(self.rear_wheel_base / (self.front_wheel_base + self.rear_wheel_base) * np.tan(wheel_angle))
        
        next_x = location[0] + speed * np.cos(heading + slip_angle) * self.time_step
        next_y = location[1] + speed * np.sin(heading + slip_angle) * self.time_step
        next_heading = heading + speed / self.rear_wheel_base * np.sin(slip_angle) * self.time_step

        # We use different polynomial models for estimating the speed if whether the ego vehicle brakes or not.
        if brake:
            speed_kph = speed * 3.6
            features = speed_kph ** np.arange(1, 8)
            next_speed_kph = features @ self.brake_values
            next_speed = next_speed_kph / 3.6
        else:
            throttle = np.clip(throttle, 0., 1.0)

            # For a throttle value < 0.3 the car doesn't really accelerate and the polynomial model below doesn't hold anymore.
            if throttle < self.throttle_threshold_during_forecasting:
                next_speed = speed
            else:
                speed_kph = speed * 3.6
                features = np.array([speed_kph,
                                    speed_kph**2,
                                    throttle,
                                    throttle**2,
                                    speed_kph * throttle,
                                    speed_kph * throttle**2,
                                    speed_kph**2 * throttle,
                                    speed_kph**2 * throttle**2]).T

                next_speed_kph = features @ self.throttle_values
                next_speed = next_speed_kph / 3.6

        next_speed = np.maximum(0.0, next_speed)
        next_location = np.array([next_x[0], next_y[0], location[2]])

        return next_location, next_heading, next_speed