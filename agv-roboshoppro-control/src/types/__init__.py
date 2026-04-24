class AGVStatus:
    def __init__(self, position, orientation, battery_level):
        self.position = position
        self.orientation = orientation
        self.battery_level = battery_level

class Command:
    def __init__(self, action, parameters):
        self.action = action
        self.parameters = parameters