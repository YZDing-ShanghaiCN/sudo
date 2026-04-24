class MotionController:
    def __init__(self, client):
        self.client = client

    def move_forward(self, distance):
        command = {"action": "move_forward", "distance": distance}
        self.client.send_command(command)

    def turn_left(self, angle):
        command = {"action": "turn_left", "angle": angle}
        self.client.send_command(command)

    def turn_right(self, angle):
        command = {"action": "turn_right", "angle": angle}
        self.client.send_command(command)

    def stop(self):
        command = {"action": "stop"}
        self.client.send_command(command)