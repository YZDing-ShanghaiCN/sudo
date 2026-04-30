import argparse
import json
import os
import glob

import numpy as np
import OpenEXR
import Imath


def read_exr_channels(file_path: str) -> dict[str, np.ndarray]:
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"EXR file not found: {file_path}")

	exr_file = OpenEXR.InputFile(file_path)
	header = exr_file.header()
	dw = header["dataWindow"]
	width = dw.max.x - dw.min.x + 1
	height = dw.max.y - dw.min.y + 1

	float_type = Imath.PixelType(Imath.PixelType.FLOAT)
	channels = list(header["channels"].keys())

	data = {}
	for channel in channels:
		raw = exr_file.channel(channel, float_type)
		arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
		data[channel] = arr

	exr_file.close()
	return data


def compute_stats(depth: np.ndarray) -> dict:
	total = int(depth.size)
	finite = np.isfinite(depth)
	valid = int(finite.sum())
	valid_ratio = float(valid / total) if total else 0.0

	if valid == 0:
		return {
			"min": None,
			"max": None,
			"mean": None,
			"median": None,
			"std": None,
			"valid_ratio": valid_ratio,
			"valid_count": valid,
			"total_count": total,
		}

	vals = depth[finite]
	return {
		"min": float(vals.min()),
		"max": float(vals.max()),
		"mean": float(vals.mean()),
		"median": float(np.median(vals)),
		"std": float(vals.std()),
		"valid_ratio": valid_ratio,
		"valid_count": valid,
		"total_count": total,
	}


def iter_exr_files(input_dir: str) -> list[str]:
	pattern = os.path.join(input_dir, "*.exr")
	return sorted(glob.glob(pattern))


def main() -> int:
	dir_name = "wait_pose"
	parser = argparse.ArgumentParser(description="Compute per-file depth statistics for EXR files.")
	parser.add_argument(
		"--input-dir",
		default=f"/home/user/Desktop/main/main2/{dir_name}/output_depth",
		help="Directory containing depth EXR files.",
	)
	parser.add_argument(
		"--output-file",
		default=f"/home/user/Desktop/main/main2/{dir_name}/depth_stats.jsonl",
		help="Output JSONL file path.",
	)
	args = parser.parse_args()

	files = iter_exr_files(args.input_dir)
	if not files:
		print(f"No EXR files found in {args.input_dir}")
		return 1

	os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

	with open(args.output_file, "w", encoding="utf-8") as f:
		for file_path in files:
			channels = read_exr_channels(file_path)
			for channel, depth in channels.items():
				stats = compute_stats(depth)
				record = {
					"file": os.path.basename(file_path),
					"channel": channel,
					"shape": [int(depth.shape[0]), int(depth.shape[1])],
					"stats": stats,
				}
				f.write(json.dumps(record, ensure_ascii=True) + "\n")

	print(f"Wrote stats for {len(files)} files to {args.output_file}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
