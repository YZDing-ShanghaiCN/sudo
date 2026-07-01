from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "capture_rgbd_orbbec_sdk.py"
SPEC = importlib.util.spec_from_file_location("capture_rgbd_orbbec_sdk", MODULE_PATH)
capture_rgbd = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = capture_rgbd
SPEC.loader.exec_module(capture_rgbd)

try:
    import numpy as np
except ImportError:
    np = None


class CaptureRgbdPathTests(unittest.TestCase):
    def test_resolve_project_path_keeps_absolute_paths(self) -> None:
        path = capture_rgbd.resolve_project_path("/tmp/rgbd")

        self.assertEqual(path, Path("/tmp/rgbd"))

    def test_resolve_project_path_anchors_relative_paths_to_project(self) -> None:
        path = capture_rgbd.resolve_project_path("outputs/rgbd")

        self.assertEqual(path, capture_rgbd.PROJECT_ROOT / "outputs/rgbd")

    def test_build_capture_paths_uses_rgb_and_depth_subdirs(self) -> None:
        paths = capture_rgbd.build_capture_paths(
            Path("/tmp/rgbd/20260611_120000_001"),
            "orbbec rgbd",
            3,
            "jpg",
        )

        self.assertEqual(paths.stem, "orbbec_rgbd_0003")
        self.assertEqual(
            paths.depth_m_npy,
            Path("/tmp/rgbd/20260611_120000_001/depth/orbbec_rgbd_0003_depth_m.npy"),
        )
        self.assertEqual(
            paths.depth_m_png,
            Path("/tmp/rgbd/20260611_120000_001/depth/orbbec_rgbd_0003_depth_m.png"),
        )
        self.assertEqual(
            paths.depth_preview_image,
            Path(
                "/tmp/rgbd/20260611_120000_001/depth/"
                "orbbec_rgbd_0003_depth_preview.jpg"
            ),
        )
        self.assertEqual(
            paths.color_image,
            Path("/tmp/rgbd/20260611_120000_001/rgb/orbbec_rgbd_0003_color.jpg"),
        )
        self.assertEqual(
            paths.metadata_json,
            Path("/tmp/rgbd/20260611_120000_001/orbbec_rgbd_0003.json"),
        )

    def test_build_capture_paths_can_disable_color(self) -> None:
        paths = capture_rgbd.build_capture_paths(
            Path("/tmp/rgbd/fixed"),
            "orbbec",
            1,
            "png",
            include_color=False,
        )

        self.assertIsNone(paths.color_image)


class CaptureRgbdArgsTests(unittest.TestCase):
    def test_default_args_are_valid(self) -> None:
        args = capture_rgbd.parse_args([])

        self.assertIsNone(args.device_index)
        self.assertEqual(args.depth_width, 1024)
        self.assertEqual(args.depth_height, 1024)
        self.assertEqual(args.depth_fps, 15)
        self.assertFalse(args.save_depth_preview)
        self.assertTrue(capture_rgbd.validate_args(args))

    def test_can_enable_depth_preview_output(self) -> None:
        args = capture_rgbd.parse_args(["--save-depth-preview"])

        self.assertTrue(args.save_depth_preview)
        self.assertTrue(capture_rgbd.validate_args(args))

    def test_viewer_args_are_valid(self) -> None:
        args = capture_rgbd.parse_args(["--viewer"])

        self.assertTrue(capture_rgbd.validate_args(args))

    def test_rejects_partial_depth_profile(self) -> None:
        args = capture_rgbd.parse_args(["--depth-width", "640"])

        with contextlib.redirect_stderr(io.StringIO()):
            valid = capture_rgbd.validate_args(args)

        self.assertFalse(valid)

    def test_accepts_sdk_default_depth_profile_when_all_depth_values_are_zero(self) -> None:
        args = capture_rgbd.parse_args(
            ["--depth-width", "0", "--depth-height", "0", "--depth-fps", "0"]
        )

        self.assertEqual(args.depth_width, 0)
        self.assertEqual(args.depth_height, 0)
        self.assertEqual(args.depth_fps, 0)
        self.assertTrue(capture_rgbd.validate_args(args))

    def test_rejects_viewer_with_check_only(self) -> None:
        args = capture_rgbd.parse_args(["--viewer", "--check-only"])

        with contextlib.redirect_stderr(io.StringIO()):
            valid = capture_rgbd.validate_args(args)

        self.assertFalse(valid)

    def test_rejects_invalid_depth_preview_max_m(self) -> None:
        args = capture_rgbd.parse_args(["--depth-preview-max-m", "0"])

        with contextlib.redirect_stderr(io.StringIO()):
            valid = capture_rgbd.validate_args(args)

        self.assertFalse(valid)

    def test_rejects_invalid_depth_png_scale_m(self) -> None:
        args = capture_rgbd.parse_args(["--depth-png-scale-m", "0"])

        with contextlib.redirect_stderr(io.StringIO()):
            valid = capture_rgbd.validate_args(args)

        self.assertFalse(valid)

    def test_rejects_invalid_min_valid_ratio(self) -> None:
        args = capture_rgbd.parse_args(["--min-valid-ratio", "1.5"])

        with contextlib.redirect_stderr(io.StringIO()):
            valid = capture_rgbd.validate_args(args)

        self.assertFalse(valid)

    def test_rejects_negative_device_index(self) -> None:
        args = capture_rgbd.parse_args(["--device-index", "-1"])

        with contextlib.redirect_stderr(io.StringIO()):
            valid = capture_rgbd.validate_args(args)

        self.assertFalse(valid)


@unittest.skipIf(np is None, "NumPy is not installed")
class CaptureRgbdDepthMathTests(unittest.TestCase):
    def test_depth_raw_to_meters_applies_scale(self) -> None:
        raw = np.array([[0, 100], [200, 300]], dtype=np.uint16)

        depth_m = capture_rgbd.depth_raw_to_meters(raw, 0.5)

        self.assertEqual(depth_m.dtype, np.float32)
        np.testing.assert_array_equal(
            depth_m,
            np.array([[0.0, 0.05], [0.1, 0.15]], dtype=np.float32),
        )

    def test_depth_stats_ignores_zero_pixels(self) -> None:
        depth_m = np.array([[0.0, 0.05], [0.1, 0.0]], dtype=np.float32)

        stats = capture_rgbd.depth_stats(depth_m)

        self.assertEqual(stats["total_pixels"], 4)
        self.assertEqual(stats["valid_pixels"], 2)
        self.assertAlmostEqual(stats["valid_ratio"], 0.5)
        self.assertAlmostEqual(stats["min_depth_m"], 0.05)
        self.assertAlmostEqual(stats["max_depth_m"], 0.1)
        self.assertAlmostEqual(stats["mean_depth_m"], 0.075)

    def test_depth_m_to_uint16_png_applies_scale_and_clips(self) -> None:
        depth_m = np.array([[-1.0, 1.234, 70.0]], dtype=np.float32)

        png_depth = capture_rgbd.depth_m_to_uint16_png(depth_m, 0.001)

        self.assertEqual(png_depth.dtype, np.uint16)
        np.testing.assert_array_equal(
            png_depth,
            np.array([[0, 1234, 65535]], dtype=np.uint16),
        )


if __name__ == "__main__":
    unittest.main()
