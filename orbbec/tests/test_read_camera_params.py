from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "read_camera_params.py"
SPEC = importlib.util.spec_from_file_location("read_camera_params", MODULE_PATH)
read_params = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = read_params
SPEC.loader.exec_module(read_params)


class DummyIntrinsic:
    fx = 500.0
    fy = 501.0
    cx = 320.0
    cy = 240.0


class DummyDistortion:
    k1 = 0.1
    k2 = -0.2
    k3 = 0.3
    k4 = 0.0
    k5 = 0.0
    k6 = 0.0
    p1 = 0.01
    p2 = -0.02


class ReadCameraParamsSerializationTests(unittest.TestCase):
    def test_intrinsic_to_dict_adds_opencv_camera_matrix(self) -> None:
        data = read_params.intrinsic_to_dict(DummyIntrinsic())

        self.assertEqual(data["fx"], 500.0)
        self.assertEqual(
            data["K"],
            [
                [500.0, 0.0, 320.0],
                [0.0, 501.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_distortion_to_dict_adds_opencv_coefficients(self) -> None:
        data = read_params.distortion_to_dict(DummyDistortion())

        self.assertEqual(
            read_params.opencv_distortion(data),
            [0.1, -0.2, 0.01, -0.02, 0.3],
        )

    def test_append_resolution_keeps_one_entry_with_same_calibration(self) -> None:
        resolutions = {}
        first = {
            "resolution": "640x480",
            "width": 640,
            "height": 480,
            "intrinsic": {"fx": 1.0},
            "distortion": {"k1": 0.0},
            "opencv_distortion_k1_k2_p1_p2_k3": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        second = {
            "resolution": "640x480",
            "width": 640,
            "height": 480,
            "intrinsic": {"fx": 1.0},
            "distortion": {"k1": 0.0},
            "opencv_distortion_k1_k2_p1_p2_k3": [0.0, 0.0, 0.0, 0.0, 0.0],
        }

        read_params.append_resolution(resolutions, first)
        read_params.append_resolution(resolutions, second)

        self.assertEqual(len(resolutions), 1)
        self.assertNotIn("variants", resolutions["640x480"])

    def test_append_resolution_keeps_variants_when_calibration_differs(self) -> None:
        resolutions = {}
        first = {
            "resolution": "640x480",
            "width": 640,
            "height": 480,
            "intrinsic": {"fx": 1.0},
            "distortion": {"k1": 0.0},
            "opencv_distortion_k1_k2_p1_p2_k3": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        second = {
            "resolution": "640x480",
            "width": 640,
            "height": 480,
            "intrinsic": {"fx": 2.0},
            "distortion": {"k1": 0.0},
            "opencv_distortion_k1_k2_p1_p2_k3": [0.0, 0.0, 0.0, 0.0, 0.0],
        }

        read_params.append_resolution(resolutions, first)
        read_params.append_resolution(resolutions, second)

        self.assertEqual(len(resolutions["640x480"]["variants"]), 2)


if __name__ == "__main__":
    unittest.main()
