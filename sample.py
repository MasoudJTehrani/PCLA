import carla
import time
from PCLA import PCLA



def main():

    HOST_IP  = "localhost"
    client = carla.Client(HOST_IP, 2000)
    client.set_timeout(10.0)
    client.load_world("Town02")
    synchronous_master = False
    pcla = None
    settings = None

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)
        
        settings = world.get_settings()
        asynch = False
        if not asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
                    you could experience some issues. If it's not working correctly, switch to \
                    synchronous mode by using traffic_manager.set_synchronous_mode(True)")
        #settings.no_rendering_mode = True
        world.apply_settings(settings)
        
        # Finding actors
        bpLibrary = world.get_blueprint_library()

        ## Finding vehicle
        vehicleBP = bpLibrary.filter('model3')[0]

        vehicle_spawn_points = world.get_map().get_spawn_points()

        ### Spawn vehicle
        vehicle = world.spawn_actor(vehicleBP, vehicle_spawn_points[31])
        
        # Retrieve the spectator object
        spectator = world.get_spectator()

        # Set the spectator with our transform
        spectator.set_transform(carla.Transform(carla.Location(x=-8, y=108, z=7), carla.Rotation(pitch=-19, yaw=0, roll=0)))

        world.tick()

        agent = "simlingo_simlingo"
        route = "./sample_route.xml"
        pcla = PCLA(agent, vehicle, route, client)
        
        print('\nSpawned the vehicle with model =', agent,', press Ctrl+C to exit.\n')
        step = 0
        while True:
            try:
                ego_action = pcla.get_action()
                vehicle.apply_control(ego_action)
                world.tick()
                step += 1
            except Exception as e:
                print(f'\nError at step {step}:')
                print(f'{type(e).__name__}: {e}\n')
                import traceback
                traceback.print_exc()
                break
    
    finally:
        #settings.no_rendering_mode = False
        if settings is not None:
            settings.synchronous_mode = False
            world.apply_settings(settings)

        # Destroy vehicle
        print('\nCleaning up the vehicle')
        if pcla is not None:
            pcla.cleanup()
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.')
