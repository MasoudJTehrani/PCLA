import cv2
import numpy as np
import math

# Visualizes the first element of a batch
def viz_batch(batch, pix_per_m=5, rgb=None, input_ego=False):
    batch_idxs = batch["idxs"].cpu().numpy()
    x_batch_objs = batch["x_objs"].cpu().numpy()
    waypoints_batch = batch["waypoints"].cpu().numpy()
    targetpoint_batch = batch["target_point"].cpu().numpy()
    objs = x_batch_objs[batch_idxs[0]]

    if not input_ego:
        objs = objs[1:] # Ego padding token

    max_d = 30 # NOTE
    if rgb is not None:
        size = rgb[1].shape[0]
        pix_per_m = size/max_d/2
    else:
        size = int(max_d*pix_per_m*2)
    origin = (size//2, size//2)
    PIXELS_PER_METER = pix_per_m

    img = np.zeros((size, size, 3), dtype=np.uint8)

    for o in objs:
        x = o[2]*PIXELS_PER_METER + origin[1] # Image x points right, so we use y axis
        y = -o[1]*PIXELS_PER_METER + origin[0] # Image y points down, so we use negative y axis
        yaw = o[3]
        extent_x = o[6]*PIXELS_PER_METER # Flip extents accordingly
        extent_y = o[5]*PIXELS_PER_METER
        origin_v = (x, y)
        vel = o[4]/3.6  # in m/s

        if o[0] == 0: # Padding
            continue
        elif o[0] in [0, 4]:  # EGO and static car
            fill_color = [50, 50, 50]
            outline_color = [127, 127, 127]
        elif o[0] == 1:  # Red for car
            fill_color = [0, 0, 100]
            outline_color = [0, 0, 255]
        elif o[0] == 2:  # Blue for route
            fill_color = [100, 0, 0]
            outline_color = [255, 0, 0]
        elif o[0] == 3:  # Cyan for walker
            fill_color = [100, 100, 0]
            outline_color = [255, 255, 0]
        elif o[0] == 6:  # Magenta w/o fill for stop sign
            fill_color = [0, 0, 0]
            outline_color = [255, 0, 255]
        elif o[0] == 7:  # Red w/o fill for traffic light
            fill_color = [0, 0, 0]
            outline_color = [0, 0, 255]
        else:
            fill_color = [127, 127, 127]
            outline_color = [255, 0, 255]

        if extent_x == 0:
            print("X == 0")
            extent_x = 3
            extent_y = 3

        box = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
        box = np.array(box).astype(int)
        draw = np.zeros_like(img)

        cv2.fillPoly(draw, [box], fill_color)
        cv2.drawContours(draw, [box], 0, outline_color, 1)

        # Creates speed indicator for vehicles and center dot for routes
        endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
        cv2.line(draw, (int(endx1), int(endy1)), (int(endx2), int(endy2)), color=outline_color, thickness=1)

        img = cv2.addWeighted(img, 1, draw, 1, 0)

    for wp in waypoints_batch[0]:
        x = wp[1]*PIXELS_PER_METER + origin[1]
        y = - wp[0]*PIXELS_PER_METER + origin[0]
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)

    target_point = targetpoint_batch[0]
    x = target_point[1]*PIXELS_PER_METER + origin[1]
    y = - target_point[0]*PIXELS_PER_METER + origin[0]
    cv2.circle(img, (int(x), int(y)), 3, (255, 0, 200), -1)

    if rgb is not None:
        img = np.hstack((img, rgb[1][..., [0, 1, 2]]))

    return img


def rotate_point(x, y, theta):
    return (x*math.cos(theta) - y*math.sin(theta),   y*math.cos(theta) + x*math.sin(theta))


def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2


def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)
