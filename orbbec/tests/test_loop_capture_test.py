from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "loop_capture_test.py"
SPEC = importlib.util.spec_from_file_location("loop_capture_test_runner", MODULE_PATH)
loop_capture = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = loop_capture
SPEC.loader.exec_module(loop_capture)


class DummyIntrinsic:
    fx = 910.5
    fy = 910.1
    cx = 638.2
    cy = 362.4


class DummyColorProfile:
    def get_width(self) -> int:
        return 1280

    def get_height(self) -> int:
        return 720

    def get_intrinsic(self) -> DummyIntrinsic:
        return DummyIntrinsic()


class LoopCaptureTestRunnerTests(unittest.TestCase):
    def test_session_name_uses_rgbd_prefix_and_two_digit_year(self) -> None:
        name = loop_capture.session_name(datetime(2026, 7, 7, 12, 34, 56))

        self.assertEqual(name, "rgbd_260707_123456")

    def test_project_and_data_directory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "results_test" / "260707_123456"
            rgb_dir, depth_dir, model_dir, recon_dir = loop_capture.create_project_dirs(
                session_dir
            )

            self.assertEqual(rgb_dir, session_dir / "rgb")
            self.assertEqual(depth_dir, session_dir / "depth")
            self.assertTrue(rgb_dir.is_dir())
            self.assertTrue(depth_dir.is_dir())
            self.assertTrue(model_dir.is_dir())
            self.assertTrue(recon_dir.is_dir())
            self.assertEqual(list(model_dir.iterdir()), [])
            self.assertEqual(list(recon_dir.iterdir()), [])

    def test_capture_args_disable_extra_metadata_files(self) -> None:
        capture_rgbd = loop_capture.load_capture_module()
        args = loop_capture.capture_args(
            capture_rgbd,
            Path("/tmp/results_test/260707_123456"),
        )

        self.assertTrue(args.viewer)
        self.assertTrue(args.no_metadata)
        self.assertEqual(args.session_name, "260707_123456")
        self.assertEqual(args.color_output, "png")
        self.assertEqual(args.depth_output, "png")

    def test_capture_paths_are_six_digit_png_names_starting_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "260707_123456"
            rgb_dir, depth_dir, _, _ = loop_capture.create_project_dirs(session_dir)
            capture_rgbd = loop_capture.load_capture_module()
            loop_capture.configure_capture_layout(
                capture_rgbd,
                session_dir,
                rgb_dir,
                depth_dir,
            )

            paths = capture_rgbd.build_capture_paths(
                session_dir,
                "ignored",
                1,
                "png",
            )

            self.assertEqual(paths.stem, "000000")
            self.assertEqual(paths.color_image, rgb_dir / "000000.png")
            self.assertEqual(paths.depth_m_png, depth_dir / "000000.png")

    def test_configuration_uses_absolute_paths_and_profile_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "260707_123456"
            _, _, model_dir, recon_dir = loop_capture.create_project_dirs(
                session_dir
            )

            configuration = loop_capture.build_configuration(
                session_dir,
                model_dir,
                recon_dir,
                DummyColorProfile(),
            )
            output_path = loop_capture.write_configuration(
                session_dir,
                configuration,
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved["projectname"], "DemoProject")
            self.assertEqual(saved["camera"]["resolution"], [1280, 720])
            self.assertEqual(
                saved["camera"]["intrinsic"],
                [
                    [910.5, 0.0, 638.2],
                    [0.0, 910.1, 362.4],
                    [0.0, 0.0, 1.0],
                ],
            )
            self.assertEqual(saved["environment"]["datasrc"], str(session_dir.resolve()))
            self.assertEqual(
                saved["environment"]["modelsrc"], str(model_dir.resolve())
            )
            self.assertEqual(
                saved["environment"]["reconstructionsrc"],
                str(recon_dir.resolve()),
            )

    def test_existing_project_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "260707_123456"
            session_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "拒绝覆盖"):
                loop_capture.create_project_dirs(session_dir)


if __name__ == "__main__":
    unittest.main()
