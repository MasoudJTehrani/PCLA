def route_maker(waypoints, savePath="route.xml"):
    # This function gets a list of carla waypoints and convert it into leaderboard route
    from xml.dom import minidom 

    if len(waypoints) <= 1:
        print("Please provide more than 1 waypoint")
        return

    root = minidom.Document()
    xml = root.createElement('route')
    xml.setAttribute('id', "_")
    xml.setAttribute('town', "_")
    root.appendChild(xml)
    
    for wp in waypoints:
        tf = wp.transform
        productChild = root.createElement('waypoint')
        productChild.setAttribute('pitch', str(tf.rotation.pitch))
        productChild.setAttribute('roll', str(tf.rotation.roll))
        productChild.setAttribute('x', str(tf.location.x))
        productChild.setAttribute('y', str(tf.location.y))
        productChild.setAttribute('yaw', str(tf.rotation.yaw))
        productChild.setAttribute('z', str(tf.location.z))
        xml.appendChild(productChild)
    
    xml_str = root.toprettyxml(indent="\t")
    
    with open(savePath, "w") as f: 
        f.write(xml_str)
