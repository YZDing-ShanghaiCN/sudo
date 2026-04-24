from src.client.roboshoppro_client import RoboshopProClient
from src.controller.motion_controller import MotionController
import time

def main():
    # Initialize the RoboshopPro client
    client = RoboshopProClient()

    # Initialize the motion controller
    motion_controller = MotionController()

    # Example commands to control the AGV
    commands = [
        {"action": "move_forward", "duration": 5},
        {"action": "turn_left", "duration": 2},
        {"action": "move_forward", "duration": 3},
        {"action": "turn_right", "duration": 2},
        {"action": "stop", "duration": 0}
    ]

    for command in commands:
        if command["action"] == "move_forward":
            motion_controller.move_forward()
            time.sleep(command["duration"])
        elif command["action"] == "turn_left":
            motion_controller.turn_left()
            time.sleep(command["duration"])
        elif command["action"] == "turn_right":
            motion_controller.turn_right()
            time.sleep(command["duration"])
        elif command["action"] == "stop":
            motion_controller.stop()

    # Optionally, send status update to the client
    client.send_status_update()

if __name__ == "__main__":
    main()