class RoboshopProClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def send_command(self, command):
        # Code to send command to the AGV via the RoboshopPro API
        pass

    def receive_status(self):
        # Code to receive status updates from the AGV
        pass

    def move_forward(self):
        self.send_command("MOVE_FORWARD")

    def turn_left(self):
        self.send_command("TURN_LEFT")

    def turn_right(self):
        self.send_command("TURN_RIGHT")

    def stop(self):
        self.send_command("STOP")