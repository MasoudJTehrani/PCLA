import carla
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from leaderboard_codes.global_route_planner import GlobalRoutePlanner
from leaderboard_codes.global_route_planner_dao import GlobalRoutePlannerDAO

def location_to_waypoint(client, starting_location, ending_location, distance=2, draw=False):
    # This function is used to generate waypoints between two locations
    world = client.get_world()
    amap = world.get_map()
    dao = GlobalRoutePlannerDAO(amap, distance)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    w1 = grp.trace_route(starting_location, ending_location)
    
    # draw the route on the carla simulator
    if draw:
        for i, w in enumerate(w1):
            color = carla.Color(r=255, g=0, b=0) if i % 10 == 0 else carla.Color(r=0, g=0, b=255)
            world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                                     color=color, life_time=60.0, persistent_lines=True)
    
    return [wp[0] for wp in w1]
