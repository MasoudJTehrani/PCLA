import svg
import numpy as np
import json

import pickle

TOWNS = {}
for town in ["Town12","Town13"]:
    
    with open(f'/home/gerstenecker/PlanT_2_cleanup/PlanT/util/{town}.pickle', 'rb') as handle:
        TOWNS[town] = pickle.load(handle)

def intersects(x, y, radius, bbox):
    xmin, xmax, ymin, ymax = bbox
    
    # Find the closest point on the bbox to the circle center
    closest_x = max(xmin, min(x, xmax))
    closest_y = max(ymin, min(y, ymax))
    
    # Calculate the distance from the circle center to this point
    distance_x = x - closest_x
    distance_y = y - closest_y
    
    distance_squared = distance_x ** 2 + distance_y ** 2
    
    # If the distance is less than or equal to radius squared, they intersect
    return distance_squared <= radius ** 2

def get_roads(x, y, yaw, town="Town13", local_extent = 64):

    roads, lane_markings, center_lines = TOWNS[town]

    radius = (2 * local_extent**2)**0.5

    elements = [road for road, bbox in roads if intersects(x, y, radius, bbox)]
    elements += [line for line, bbox in lane_markings if intersects(x, y, radius, bbox)]
    elements += [line for line, bbox in center_lines if intersects(x, y, radius, bbox)]

    # Inverse transform into view
    return svg.G(elements=elements, transform=[svg.Scale(10, 10), svg.Rotate(yaw, local_extent, local_extent), svg.Translate(-y+local_extent, x+local_extent)])
# static border 48c776
# static ("#c9e4de", "#60a67a")
# static ("#faedcb", "#f7ba2a")

# weißes ego ("#edebeb", "#f7f5f5")

colors = [None, ("#b3b3b5", "#7a7a7a"), ("#d0e5f5", "#6da5f2"), ("#dbcdf0", "#a677ed"), ("#fad9af", "#e09634"), ("#e7ddf0", "#8a60d6"), ("#ebc7c7", "#e35454"), ("#f2c6de", "#d979ad")]
# colors = [None, ("#dad7db", "#aeabb0"), ("#8fd6ff", "#65a9fc"), ("#ce8fff", "#5a1675"), ("#ffce8f", "#f79e2a"), ("#dadee6", "#d419d4"), ("#dadee6", "#d419d4"), ("#dadee6", "#d419d4")]
ego_box = [1, 0, 0, 0, 0, 0.9183566570281982*2, 2.44619083404541*2]

def box2svg(box, color=None, nospeed=False):
    c, y, x, yaw, speed, w, h = box
    c = int(c)
    x = x*10 + 640
    y = - y*10 + 640
    w *= 10
    h *= 10
    speed *= 10/4
    
    speed_stroke = 2.7
    speed = max(speed_stroke/2, speed)

    box_stroke = 2.7

    if color is None:
        fill_color, line_color = colors[c]
    else:
        fill_color, line_color = color

    box = svg.Rect(
                x=-w/2, y=-h/2,
                rx=w/5,
                width=w, height=h,
                stroke=line_color,
                fill=fill_color,
                fill_opacity=1,
                stroke_width=box_stroke,
                transform_origin="center" # TODO des kann weg oder
            )
    
    g = svg.G(elements=[box, speed], transform=[svg.Translate(x, y), svg.Rotate(yaw, 0, 0)])

    if c not in [1] and not nospeed:
        g.elements.append(
            svg.Line(
                # x1=w/2,
                # y1=h/2 + speed_stroke/2,
                # x2=w/2,
                # y2=h/2-speed,
                x1 = 0,
                y1 = speed_stroke/2,
                x2 = 0,
                y2 = -speed,
                stroke=line_color,
                stroke_width=speed_stroke,))

    

    return g

def waypoint(p, color, secondary = None, opacity=1, radius=3):
    y, x = p
    x = x*10 + 640 #  + ego_box[5]*5
    y = - y*10 + 640 # + ego_box[6]*5

    stroke_width = 0 if secondary is None else radius/3
    stroke = secondary

    return svg.Circle(cx=x, cy=y, r = radius, fill=color, stroke_width=stroke_width, stroke=stroke, opacity=opacity)

def viz_elements(boxes, route_original, route, waypoints):
    elements = [box2svg(ego_box)]
    elements += [waypoint(x, "#1f49f0", "#183299", 0.5) for x in route_original]
    elements += [box2svg(box) for box in boxes]
    elements += [waypoint(x, "#eb0e0e", "#ab0303", 1) for x in route]
    elements += [waypoint(x, "#00cf1f", "#05961a", radius=3.5) for x in waypoints]
    return elements

def control_indicator(controls):
    w = 15
    max_h = 50
    line_colors = {False: "#4527db", True: "#db2323"}
    fill_colors = {False: "#7760e6", True: "#f06e6e"}
    stroke = 2

    throttle = 0.5
    brake = True

    elements = []
    for i, (_, throttle, brake) in enumerate(controls):
        height = max(1, max_h * throttle)
        brake = bool(brake)
        if brake:
            throttle = 0
            height = max_h/4

        x_offset = i*w + i*(stroke+1) + 2
        y_offset = max_h * (1 - throttle) + 2

        elements.append(svg.Rect(
                x=x_offset + stroke/2, y=y_offset + stroke/2,
                width=w, height=height,
                stroke=line_colors[brake],
                fill=fill_colors[brake],
                fill_opacity=1,
                stroke_width=stroke
            ))

    return elements

def make_animation(txt_file, out_file, town, center_frame, duration=40, control=True, fps=3):
    with open(txt_file) as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines]
    records = [x for x in records if abs(x["frame"]-center_frame) < duration/2]

    # w = 1280
    # h = 1280
    w = 640
    h = 640

    # w = 320
    # h = 320
    OFFSET_X = 50

    canvas = svg.SVG(
        width=w,
        height=h,
        elements=[]
    )

    

    for i, record in enumerate(records):
        g = svg.G(id=f"frame{i}", class_="frame", elements=[], opacity=0, transform=[svg.Translate((w-1280)/2, (h-1280)/2+ OFFSET_X)])
        if town!="":
            g.elements.append(get_roads(*record["ego_pos"], -np.rad2deg(record["ego_rot"]), town))
        g.elements += viz_elements(record["boxes"], [], record["route"], record["waypoints"]) #record["route_original"], record["route"], record["waypoints"]) #  
        if control:
            g.elements += control_indicator(record["control_history"])
        canvas.elements.append(g)

    frames = len(canvas.elements)

    animation = f"""@keyframes pulse {{
    0% {{ opacity: 1; }}
    {100/frames}% {{ opacity: 0; }}
}}

.frame {{
    animation: pulse {frames/fps}s infinite steps(1) forwards;
}}
"""

    for frame in range(frames):
        animation += f"#frame{frame} {{animation-delay: {frame/fps}s}}"

    canvas.elements.append(svg.Style(text=animation))

    with open(out_file, "w") as f:
        f.write(str(canvas))