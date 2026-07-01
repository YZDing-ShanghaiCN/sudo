# PoseEstimation

This repository hosts multiple pose estimation models. Each model lives in its own
subdirectory (for example: `megapose6d/`) and documents its own usage details.

## Environment

Environment requirements vary by model. In general you will need:

- Python and common ML dependencies
- Optional GPU + CUDA for accelerated inference
- Model-specific packages listed in each model README

Always follow the README inside the model directory for the exact setup.

## Inputs

Typical inputs for pose estimation models include:

- RGB or RGB-D images
- Camera intrinsics (and optional extrinsics)
- Object list / detections
- Optional CAD or mesh assets for target objects

Each model defines the exact input format in its own README.

## Outputs

Typical outputs include:

- 6D pose results (often JSON or CSV)
- Optional visualization images
- Optional logs and analysis summaries

Check the model README for the concrete output paths and file formats.
