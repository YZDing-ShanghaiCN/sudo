import unittest
from src.controller.motion_controller import MotionController

class TestMotionController(unittest.TestCase):

    def setUp(self):
        self.controller = MotionController()

    def test_move_forward(self):
        initial_position = self.controller.get_position()
        self.controller.move_forward()
        new_position = self.controller.get_position()
        self.assertNotEqual(initial_position, new_position)

    def test_turn_left(self):
        initial_direction = self.controller.get_direction()
        self.controller.turn_left()
        new_direction = self.controller.get_direction()
        self.assertNotEqual(initial_direction, new_direction)

    def test_turn_right(self):
        initial_direction = self.controller.get_direction()
        self.controller.turn_right()
        new_direction = self.controller.get_direction()
        self.assertNotEqual(initial_direction, new_direction)

    def test_stop(self):
        self.controller.move_forward()
        self.controller.stop()
        self.assertEqual(self.controller.get_speed(), 0)

if __name__ == '__main__':
    unittest.main()