# Standard Library
import argparse
import json
import traceback
from pathlib import Path
from typing import List, Set

# Third Party
import numpy as np
from PIL import Image

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)


def load_observation_tensor_from_image(
    image_path: Path,
    camera_data_path: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    if load_depth:
        raise ValueError("Depth input is not supported for image_dir inference.")

    camera_data = CameraData.from_json(camera_data_path.read_text())
    rgb = np.array(Image.open(image_path), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    observation = ObservationTensor.from_numpy(rgb, None, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(example_dir: Path) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs" / "object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def pose_estimates_to_json(
    image_name: str,
    pose_estimates: PoseEstimatesType,
) -> List[dict]:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    records = []
    for label, pose in zip(labels, poses):
        object_data = ObjectData(label=label, TWO=Transform(pose)).to_json()
        object_data["image_name"] = image_name
        object_data["success"] = True
        records.append(object_data)
    return records


def collect_image_paths(image_dir: Path, suffixes: Set[str]) -> List[Path]:
    image_paths = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    return sorted(image_paths, key=lambda p: p.name)


def validate_inputs(
    example_dir: Path,
    image_dir: Path,
    camera_data_path: Path,
    input_object_data_path: Path,
    meshes_dir: Path,
) -> None:
    if not example_dir.exists():
        raise FileNotFoundError(f"example_dir not found: {example_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not camera_data_path.exists():
        raise FileNotFoundError(f"camera_data.json not found: {camera_data_path}")
    if not input_object_data_path.exists():
        raise FileNotFoundError(
            f"inputs/object_data.json not found: {input_object_data_path}"
        )
    if not meshes_dir.exists():
        raise FileNotFoundError(f"meshes dir not found: {meshes_dir}")


def run_inference_on_image_dir(
    example_dir: Path,
    image_dir: Path,
    model_name: str,
) -> None:
    model_info = NAMED_MODELS[model_name]

    camera_data_path = example_dir / "camera_data.json"
    input_object_data_path = example_dir / "inputs" / "object_data.json"
    meshes_dir = example_dir / "meshes"

    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    validate_inputs(
        example_dir,
        image_dir,
        camera_data_path,
        input_object_data_path,
        meshes_dir,
    )
    image_paths = collect_image_paths(image_dir, suffixes)
    if not image_paths:
        raise FileNotFoundError(f"no images found in: {image_dir}")

    if model_info["requires_depth"]:
        raise ValueError(
            f"Model {model_name} requires depth input, but image_dir provides RGB only."
        )

    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    logger.info(f"Running inference on {len(image_paths)} images.")
    results = []
    for idx, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing {idx}/{len(image_paths)}: {image_path.name}")
        try:
            observation = load_observation_tensor_from_image(
                image_path,
                camera_data_path,
                load_depth=model_info["requires_depth"],
            ).cuda()
            output, _ = pose_estimator.run_inference_pipeline(
                observation, detections=detections, **model_info["inference_parameters"]
            )
            results.extend(pose_estimates_to_json(image_path.name, output))
            logger.info(f"Success: {image_path.name}")
        except Exception:
            error_message = traceback.format_exc()
            results.append(
                {
                    "image_name": image_path.name,
                    "success": False,
                    "error": error_message,
                }
            )
            logger.exception(f"Failed: {image_path.name}")

    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(json.dumps(results))
    logger.info(f"Wrote predictions: {output_fn}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name", nargs="?")
    parser.add_argument("--example-dir", type=str)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    return parser.parse_args()


def resolve_example_dir(args: argparse.Namespace) -> Path:
    if args.example_dir:
        return Path(args.example_dir)
    if args.example_name:
        return LOCAL_DATA_DIR / "examples" / args.example_name
    raise ValueError("Provide example_name or --example-dir.")


def resolve_image_dir(example_dir: Path, image_dir_arg: str) -> Path:
    image_dir = Path(image_dir_arg)
    if not image_dir.is_absolute():
        image_dir = example_dir / image_dir
    return image_dir


if __name__ == "__main__":
    set_logging_level("info")
    args = parse_args()
    example_dir = resolve_example_dir(args)
    image_dir = resolve_image_dir(example_dir, args.image_dir)
    run_inference_on_image_dir(example_dir, image_dir, args.model)
