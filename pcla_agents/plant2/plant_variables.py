class PlanTVariables:
    bev_colors = [[0.485, 0.456, 0.406],# Background: Imagenet mean
                [0.25, 0.25, 0.75], # Street: Blue
                [0.485, 0.456, 0.406],# Sidewalk: Imagenet mean
                [0.75, 0.25, 0.25], # All lines: Red
                [0.25, 0.75, 0.25]] # Broken lines: Green
    
    speed_cats = {50: 0, 80: 1, 100: 2, 120: 3}

    class_nums = {# "ego_car": 1.0,
                    "car": 1.0,
                    "walker": 2.0,
                    "static": 3.0,
                    # "static_trafficwarning": 3.0,
                    "static_car": 1.0,
                    "stop_sign": 4.0,
                    "traffic_light": 5.0,
                    "emergency": 6.0
                 }
    
    car_types = ["car", "walker","emergency"]

    target_speeds = [0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20]