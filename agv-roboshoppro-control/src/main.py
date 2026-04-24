import yaml
from client.roboshoppro_client import RoboshopProClient
from controller.motion_controller import MotionController

def load_config():
    with open('config/map.yaml', 'r') as map_file:
        map_config = yaml.safe_load(map_file)
    with open('config/robot.yaml', 'r') as robot_file:
        robot_config = yaml.safe_load(robot_file)
    return map_config, robot_config

def main():
    map_config, robot_config = load_config()
    
    client = RoboshopProClient(map_config['smap_path'])
    motion_controller = MotionController(robot_config)

    # Example commands to control the AGV
    motion_controller.move_forward()
    motion_controller.turn_left()
    motion_controller.move_forward()
    motion_controller.stop()

if __name__ == "__main__":
    main()