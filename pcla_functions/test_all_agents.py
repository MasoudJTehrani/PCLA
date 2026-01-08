import os
import sys
import json
import carla
import time
import traceback
from datetime import datetime
from io import StringIO

# Set CUBLAS workspace config before any torch imports (required by some agents like carl)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Disable DeepSpeed custom op builds (avoids nvcc requirement on hosts without CUDA toolkit)
os.environ.setdefault('DS_BUILD_OPS', '0')

# Disable Weights & Biases in non-interactive test runs to avoid login prompts
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_SILENT'] = 'true'
os.environ.setdefault('WANDB_MODE', 'offline')

# Setup path early so we can import PCLA
pcla_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pcla_dir not in sys.path:
    sys.path.insert(0, pcla_dir)

def load_agents_config(pcla_dir):
    """Load agents.json and return (agent_list, total_count, skipped_count, skipped_names)."""
    config_path = os.path.join(pcla_dir, "agents.json")
    with open(config_path, 'r') as f:
        agents_config = json.load(f)
    
    # Agents to skip (missing deps or heavy models you don't want to load now)
    skip_agents = {'if_if', 'simlingo_simlingo'}
    # tfv3 currently excluded due to mmcv-full build issues on this environment
    skip_families = {'lmdrive', 'tfv3'}
    
    agent_list = []
    total_agents = 0
    skipped_agents = 0
    skipped_names = []
    # Seeded agents: only apply seeds to specific agent families
    seeded_families = {
        'carl': ('plant', 'roach', 'carl'),
        'tfv4': ('l6', 'lav', 'wp', 'aim'),
    }
    
    for agent_family, variants in agents_config.items():
        # Skip entire families (e.g., lmdrive) if requested
        family_skipped = agent_family in skip_families
        for variant_name, variant_config in variants.items():
            # Check if this family/variant combo needs a seed
            seed_suffix = ''
            if agent_family in seeded_families and variant_name in seeded_families[agent_family]:
                seed_suffix = '_0'  # Use only seed 0
            
            agent_full_name = f"{agent_family}_{variant_name}{seed_suffix}"
            total_agents += 1
            
            # Skip known incompatible agents
            if family_skipped or agent_full_name in skip_agents:
                skipped_agents += 1
                skipped_names.append(agent_full_name)
                continue
            
            agent_list.append(agent_full_name)
    
    return agent_list, total_agents, skipped_agents, skipped_names

def test_agent(agent_name, pcla_dir, world, vehicle, route_path, client):
    """
    Test a single agent: initialize, run 20 frames, return True if passed.
    Returns (passed: bool, error_msg: str or None)
    """
    try:
        import torch
        # Reset torch dtype to default (some agents like neat may set bfloat16)
        torch.set_default_dtype(torch.float32)
        
        # Reset deterministic algorithms (carl sets True, causing cumsum issues in tfv4)
        torch.use_deterministic_algorithms(False)
        
        # Mock stdin for agents that prompt for user input (e.g., LAV fast)
        # Supply "3" as the default choice
        old_stdin = sys.stdin
        sys.stdin = StringIO("3\n")
        
        try:
            # Import PCLA (sys.path already set at module level)
            from PCLA import PCLA
            
            pcla = PCLA(agent_name, vehicle, route_path, client)
            
            # Run 20 frames
            for frame in range(20):
                try:
                    ego_action = pcla.get_action()
                    vehicle.apply_control(ego_action)
                    world.tick()
                except Exception as e:
                    pcla.cleanup()
                    return False, f"Frame {frame}: {str(e)}"
            
            pcla.cleanup()
            return True, None
        finally:
            sys.stdin = old_stdin
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg

def main():
    # Setup CARLA
    global pcla_dir
    
    HOST_IP = "localhost"
    # Change to PCLA directory so relative paths in agent configs work
    os.chdir(pcla_dir)
    
    client = carla.Client(HOST_IP, 2000)
    client.set_timeout(10.0)
    client.load_world("Town02")
    
    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)
        
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Vehicle & spectator setup helpers
        bpLibrary = world.get_blueprint_library()
        vehicleBP = bpLibrary.filter('model3')[0]
        vehicle_spawn_points = world.get_map().get_spawn_points()
        spectator = world.get_spectator()

        def spawn_vehicle():
            # Try preferred spawn, then fall back through the list
            vehicle_candidates = [31] + list(range(len(vehicle_spawn_points)))
            veh = None
            for idx in vehicle_candidates:
                try:
                    veh = world.try_spawn_actor(vehicleBP, vehicle_spawn_points[idx])
                except IndexError:
                    continue
                if veh is not None:
                    break
            if veh is None:
                raise RuntimeError("Failed to spawn vehicle")
            # Give the sim a moment to stabilize to avoid attachment errors
            time.sleep(0.5)
            spectator.set_transform(carla.Transform(carla.Location(x=-8, y=108, z=7), carla.Rotation(pitch=-19, yaw=0, roll=0)))
            world.tick()
            return veh

        vehicle = spawn_vehicle()
        
        # Load all agents
        agent_list, total_agents, skipped_agents, skipped_names = load_agents_config(pcla_dir)
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Testing {len(agent_list)} agents (total {total_agents}, skipping {skipped_agents})...")
        if skipped_names:
            print(f"Skipped: {', '.join(skipped_names)}")
        print(f"{'='*70}\n")
        
        route_path = os.path.join(pcla_dir, "sample_route.xml")
        
        for i, agent_name in enumerate(agent_list, 1):
            print(f"[{i}/{len(agent_list)}] Testing: {agent_name}")

            # Ensure a fresh vehicle for each agent (previous cleanup may destroy it)
            if vehicle is None or not vehicle.is_alive:
                vehicle = spawn_vehicle()

            passed, error_msg = test_agent(agent_name, pcla_dir, world, vehicle, route_path, client)
            
            # PCLA.cleanup may destroy the vehicle; reset reference
            vehicle = None

            if passed:
                results[agent_name] = "passed"
                print(f"  ✓ PASSED")
            else:
                results[agent_name] = error_msg
                print(f"  ✗ FAILED: {error_msg.split(chr(10))[0][:80]}")
                print(f"  Skipping to next agent...")
            
            print()
        
        # Cleanup vehicle (if any remains)
        try:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        except Exception:
            pass

        # Write results to file
        documents_dir = os.path.join(pcla_dir, "documents")
        os.makedirs(documents_dir, exist_ok=True)
        results_file = os.path.join(documents_dir, "agent_test_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Agent Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            passed_count = sum(1 for v in results.values() if v == "passed")
            failed_count = len(results) - passed_count
            
            f.write(f"Summary: {passed_count} passed, {failed_count} failed out of {len(results)} agents\n\n")
            f.write(f"Skipped agents: {skipped_agents}\n")
            if skipped_names:
                f.write("Skipped list: " + ", ".join(skipped_names) + "\n")
            f.write(f"{'='*70}\n\n")
            
            f.write("PASSED AGENTS:\n")
            f.write("-" * 70 + "\n")
            for agent_name, status in results.items():
                if status == "passed":
                    f.write(f"{agent_name}\n")
            
            f.write(f"\n{'='*70}\n\n")
            f.write("FAILED AGENTS:\n")
            f.write("-" * 70 + "\n")
            for agent_name, error_msg in results.items():
                if error_msg != "passed":
                    f.write(f"\n{agent_name}:\n")
                    f.write(f"{error_msg}\n")
                    f.write("-" * 70 + "\n")
        
        print(f"\nResults saved to: {results_file}")
        
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
