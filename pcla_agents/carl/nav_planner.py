"""
Some helpful classes for planning and control for the privileged autopilot
"""
import carla


class RoutePlanner(object):
  """
    Gets the next waypoint along a path
    """

  def __init__(self):
    self.route = []
    self.windows_size = 2

  def set_route(self, global_plan):
    self.route = global_plan
    self.index = 0
    self.route_length = len(self.route)

  def run_step(self, gps):

    location = carla.Location(x=gps[0], y=gps[1])

    for index in range(self.index, min(self.index + self.windows_size + 1, self.route_length)):
      # Get the dot product to know if it has passed this location
      route_transform = self.route[index]
      route_location = route_transform[0].location
      wp_dir = route_transform[0].get_forward_vector()  # Waypoint's forward vector
      wp_veh = location - route_location  # vector route - vehicle

      if wp_veh.dot(wp_dir) > 0:
        self.index = index

    output = self.route[self.index:self.route_length]

    return output
