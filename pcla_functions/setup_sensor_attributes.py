def setup_sensor_attributes(bp, sensor_spec):
    if sensor_spec['type'].startswith('sensor.camera'):
        bp.set_attribute('image_size_x', str(sensor_spec['width']))
        bp.set_attribute('image_size_y', str(sensor_spec['height']))
        bp.set_attribute('fov', str(sensor_spec['fov']))
        bp.set_attribute('lens_circle_multiplier', str(3.0))
        bp.set_attribute('lens_circle_falloff', str(3.0))
        bp.set_attribute('chromatic_aberration_intensity', str(0.5))
        bp.set_attribute('chromatic_aberration_offset', str(0))
    elif sensor_spec['type'].startswith('sensor.lidar'):
        bp.set_attribute('range', str(85))
        bp.set_attribute('rotation_frequency', str(10))
        bp.set_attribute('channels', str(64))
        bp.set_attribute('upper_fov', str(10))
        bp.set_attribute('lower_fov', str(-30))
        bp.set_attribute('points_per_second', str(600000))
        bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
        bp.set_attribute('dropoff_general_rate', str(0.45))
        bp.set_attribute('dropoff_intensity_limit', str(0.8))
        bp.set_attribute('dropoff_zero_intensity', str(0.4))
    elif sensor_spec['type'].startswith('sensor.other.radar'):
        bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
        bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
        bp.set_attribute('points_per_second', '1500')
        bp.set_attribute('range', '100')   # meters
    elif sensor_spec['type'].startswith('sensor.other.gnss'):
        bp.set_attribute('noise_alt_stddev', str(0.000005))
        bp.set_attribute('noise_lat_stddev', str(0.000005))
        bp.set_attribute('noise_lon_stddev', str(0.000005))
        bp.set_attribute('noise_alt_bias', str(0.0))
        bp.set_attribute('noise_lat_bias', str(0.0))
        bp.set_attribute('noise_lon_bias', str(0.0))
    elif sensor_spec['type'].startswith('sensor.other.imu'):
        bp.set_attribute('noise_accel_stddev_x', str(0.001))
        bp.set_attribute('noise_accel_stddev_y', str(0.001))
        bp.set_attribute('noise_accel_stddev_z', str(0.015))
        bp.set_attribute('noise_gyro_stddev_x', str(0.001))
        bp.set_attribute('noise_gyro_stddev_y', str(0.001))
        bp.set_attribute('noise_gyro_stddev_z', str(0.001))
    return bp