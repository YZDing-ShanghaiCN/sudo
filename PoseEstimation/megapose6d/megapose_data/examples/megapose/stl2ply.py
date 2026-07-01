#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


ALLOWED_INPUT_SUFFIXES = {".stl", ".ply", ".obj"}
DEFAULT_TARGET_VERTICES = 5000
DEFAULT_AUTO_EDGE_DIVISOR = 100.0
AUTO_EDGE_SHRINK = 0.7
AUTO_MAX_ITERS = 12
ABS_TOL = 1e-6
REL_TOL = 1e-6


def die(message):
	print(f"Error: {message}", file=sys.stderr)
	sys.exit(1)


def warn(message):
	print(f"Warning: {message}", file=sys.stderr)


def fmt_array(arr):
	return np.array2string(
		np.asarray(arr),
		precision=6,
		separator=", ",
		suppress_small=False,
	)


def collect_stats(mesh):
	return {
		"vertices": int(len(mesh.vertices)),
		"faces": int(len(mesh.faces)),
		"bounds": np.asarray(mesh.bounds, dtype=float),
		"centroid": np.asarray(mesh.centroid, dtype=float),
		"extents": np.asarray(mesh.extents, dtype=float),
		"watertight": bool(mesh.is_watertight),
	}


def max_abs_diff(a, b):
	return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def ensure_parent_dir(path):
	parent = Path(path).parent
	if parent and not parent.exists():
		parent.mkdir(parents=True, exist_ok=True)


def load_mesh(path):
	return trimesh.load(path, force="mesh", process=False)


def parse_bool(value):
	if value is None:
		return True
	text = str(value).strip().lower()
	if text in {"1", "true", "yes", "y", "on"}:
		return True
	if text in {"0", "false", "no", "n", "off"}:
		return False
	raise argparse.ArgumentTypeError("--binary expects a boolean value")


def subdivide_to_max_edge(mesh, max_edge):
	vertices, faces = trimesh.remesh.subdivide_to_size(
		mesh.vertices,
		mesh.faces,
		max_edge=max_edge,
	)
	return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def auto_subdivide(mesh, target_vertices):
	longest = float(np.max(mesh.extents))
	if longest <= 0:
		die("Mesh extents are zero; cannot estimate max_edge.")

	max_edge = longest / DEFAULT_AUTO_EDGE_DIVISOR
	if max_edge <= 0:
		die("Estimated max_edge is invalid.")

	dense = subdivide_to_max_edge(mesh, max_edge)
	for _ in range(AUTO_MAX_ITERS):
		if len(dense.vertices) >= target_vertices:
			return max_edge, dense
		max_edge *= AUTO_EDGE_SHRINK
		dense = subdivide_to_max_edge(mesh, max_edge)

	return max_edge, dense


def tolerance_limit(scale):
	return max(ABS_TOL, REL_TOL * max(1.0, scale))


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Densify a low-poly mesh by triangle subdivision without smoothing, "
			"simplification, scaling, transforms, or repair."
		),
	)
	parser.add_argument("--input", required=True, help="Input mesh (.stl/.ply/.obj).")
	parser.add_argument("--output", required=True, help="Output .ply path.")
	parser.add_argument(
		"--max_edge",
		type=float,
		default=None,
		help="Maximum edge length for subdivision (auto if omitted).",
	)
	parser.add_argument(
		"--target_vertices",
		type=int,
		default=DEFAULT_TARGET_VERTICES,
		help="Target vertex count when max_edge is auto-estimated.",
	)
	parser.add_argument(
		"--binary",
		nargs="?",
		const=True,
		default=True,
		type=parse_bool,
		help="Save binary PLY (default: true). Use --binary false for ASCII.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	input_path = Path(args.input)
	output_path = Path(args.output)

	if not input_path.exists():
		die(f"Input file does not exist: {input_path}")
	if not input_path.is_file():
		die(f"Input path is not a file: {input_path}")
	if input_path.suffix.lower() not in ALLOWED_INPUT_SUFFIXES:
		die("Input must be .stl, .ply, or .obj.")
	if output_path.suffix.lower() != ".ply":
		die("Output must have a .ply extension.")
	if args.max_edge is not None and args.max_edge <= 0:
		die("--max_edge must be positive.")
	if args.target_vertices <= 0:
		die("--target_vertices must be positive.")

	mesh = load_mesh(input_path)
	if mesh is None or mesh.is_empty:
		die(f"Failed to load mesh from: {input_path}")
	if mesh.faces is None or len(mesh.faces) == 0:
		die("Input mesh has no faces to subdivide.")

	print("Units are preserved (e.g., mm stays mm).")
	print(
		"Surface subdivision only (no smoothing/simplify/scale/transform/repair)."
	)

	in_stats = collect_stats(mesh)

	if args.max_edge is not None:
		used_max_edge = args.max_edge
		dense_mesh = subdivide_to_max_edge(mesh, used_max_edge)
	else:
		used_max_edge, dense_mesh = auto_subdivide(mesh, args.target_vertices)

	if dense_mesh is None or dense_mesh.is_empty:
		die("Subdivision produced an empty mesh.")

	out_stats = collect_stats(dense_mesh)

	bounds_diff = max_abs_diff(in_stats["bounds"], out_stats["bounds"])
	centroid_diff = max_abs_diff(in_stats["centroid"], out_stats["centroid"])
	extents_diff = max_abs_diff(in_stats["extents"], out_stats["extents"])

	print(f"Input vertices / faces : {in_stats['vertices']} / {in_stats['faces']}")
	print(f"Output vertices / faces: {out_stats['vertices']} / {out_stats['faces']}")
	print(f"Input bounds  : {fmt_array(in_stats['bounds'])}")
	print(f"Output bounds : {fmt_array(out_stats['bounds'])}")
	print(f"Bounds max abs diff: {bounds_diff:.6g}")
	print(f"Input centroid : {fmt_array(in_stats['centroid'])}")
	print(f"Output centroid: {fmt_array(out_stats['centroid'])}")
	print(f"Input extents  : {fmt_array(in_stats['extents'])}")
	print(f"Output extents : {fmt_array(out_stats['extents'])}")
	print(f"Input watertight : {in_stats['watertight']}")
	print(f"Output watertight: {out_stats['watertight']}")
	print(f"Used max_edge: {used_max_edge:.6g}")

	scale = float(np.max(in_stats["extents"]))
	limit = tolerance_limit(scale)

	if bounds_diff > limit:
		warn("Bounds difference exceeds tolerance.")
	if centroid_diff > limit:
		warn("Centroid difference exceeds tolerance.")
	if extents_diff > limit:
		warn("Extents difference exceeds tolerance.")

	ensure_parent_dir(output_path)
	encoding = "binary_little_endian" if args.binary else "ascii"
	dense_mesh.export(output_path, file_type="ply", encoding=encoding)

	print(f"Saved dense PLY: {output_path}")
	if bounds_diff > limit:
		warn("Bounds changed noticeably; verify units and input mesh.")


if __name__ == "__main__":
	main()
