import cv2
import numpy as np
import math


def rotate_point(x, y, theta):
    return (x*math.cos(theta) - y*math.sin(theta),   y*math.cos(theta) + x*math.sin(theta))


def viz_batch(batch, index = 0, pix_per_m=5, rgb=None, range_front=128, range_sides=64, ellipse=True):
    batch_idxs = batch["idxs"]# .cpu().numpy()
    x_batch_objs = batch["x_objs"]# .cpu().numpy()
    route_batch = batch["route_original"]# .cpu().numpy()
    waypoints_batch = batch["waypoints"]# .cpu().numpy()
    pred_path = batch["pred_path"]
    objs = x_batch_objs[batch_idxs[index]]


    if rgb is not None:
        size_y = rgb[1].shape[0]
        pix_per_m = size_y/(range_front+range_sides)
        size_x = int(pix_per_m*range_sides*2)
    else:
        size_x = int(range_sides*pix_per_m*2)
        size_y = int((range_sides+range_front)*pix_per_m)
    origin = [size_y-size_x//2, size_x//2]
    PIXELS_PER_METER = pix_per_m

    # # Double front range:
    # origin[0] += size//2
    # img = np.zeros((int(size*1.5), size, 3), dtype=np.uint8)

    img = np.zeros((size_y, size_x, 3), dtype=np.uint8)
    img = cv2.ellipse(img, origin[::-1], (int(PIXELS_PER_METER*range_sides), int(PIXELS_PER_METER*range_front)), 0, 180, 360, (50, 50, 50), 3)
    img = cv2.ellipse(img, origin[::-1], (int(PIXELS_PER_METER*range_sides), int(PIXELS_PER_METER*range_sides)), 0, 0, 180, (50, 50, 50), 3)

    if "BEV" in batch:
        bev = np.array(batch["BEV"][index].permute(1, 2, 0) * 100, dtype=np.uint8)
        side = bev.shape[0]
        reshaped = int(side * PIXELS_PER_METER / 2)
        bev = cv2.resize(bev, (reshaped, reshaped))
        top = origin[0]-reshaped//2
        left = origin[1]-reshaped//2
        img[top:top+reshaped, left:left+reshaped] = bev

    # route first
    for rp in route_batch[index]:
        x = rp[1]*PIXELS_PER_METER + origin[1]  # TODO
        y = - rp[0]*PIXELS_PER_METER + origin[0]
        cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)

    for i, o in enumerate(objs):
        x = o[2]*PIXELS_PER_METER + origin[1]
        y = -o[1]*PIXELS_PER_METER + origin[0]
        yaw = o[3]
        extent_x = o[5]*PIXELS_PER_METER/2
        extent_y = o[6]*PIXELS_PER_METER/2
        origin_v = (x, y)
        vel = o[4]/3.6  # in m/s

        if o[0] == 0: # Padding
            continue
        elif o[0] == 1:  # Red for car
            fill_color = [0, 0, 100]
            outline_color = [0, 0, 255]
        elif o[0] == 2:  # Cyan for walker
            fill_color = [100, 100, 0]
            outline_color = [255, 255, 0]
        elif o[0] == 3:  # EGO and static car
            fill_color = [50, 50, 50]
            outline_color = [127, 127, 127]
        elif o[0] == 4:  # Magenta box for stop
            fill_color = [0, 0, 0]
            outline_color = [255, 0, 255]
        elif o[0] == 5:
            fill_color = [0, 0, 0]  # RED box for traffic light
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

    for wp in pred_path[index]:
        x = wp[1]*PIXELS_PER_METER + origin[1]  # TODO
        y = - wp[0]*PIXELS_PER_METER + origin[0]
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    for wp in waypoints_batch[index]:
        x = wp[1]*PIXELS_PER_METER + origin[1]  # TODO
        y = - wp[0]*PIXELS_PER_METER + origin[0]
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

    if rgb is not None:
        img = np.hstack((img, rgb[1][..., [0, 1, 2]]))

    return img


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
