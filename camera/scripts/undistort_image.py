#!/usr/bin/env python3
"""
Undistort an image using calibration data from a JSON file.
Saves the undistorted image to the original image folder.

python camera/scripts/undistort_image.py -i /path/to/image.jpg -j /home/hillbot/Downloads/fisheye/intrinsic_camera_2.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.image_preprocess import undistort_fisheye_full_view, undistort_image


def load_calibration_json(json_path):
    """Load calibration data from JSON file."""
    with open(json_path, 'r') as f:
        calib_data = json.load(f)
    
    # Extract camera matrix
    camera_matrix = calib_data['camera_matrix']['matrix']
    
    # Extract distortion coefficients
    # The JSON stores them as individual values and as a coefficients array
    distortion_coeffs = calib_data['distortion_coefficients']['coefficients']
    # Flatten the nested list structure [[k1], [k2], [k3], [k4]] -> [k1, k2, k3, k4]
    distortion_coeffs = [coeff[0] for coeff in distortion_coeffs]
    
    model = calib_data.get('model', 'fisheye')
    
    return camera_matrix, distortion_coeffs, model


def undistort_and_save(image_path, json_path):
    """
    Undistort an image using calibration data and save to original folder.
    
    Args:
        image_path: Path to the input image
        json_path: Path to the calibration JSON file
    """
    # Load calibration data
    print(f"Loading calibration data from: {json_path}")
    camera_matrix, distortion_coeffs, model = load_calibration_json(json_path)
    
    print(f"Calibration model: {model}")
    print(f"Camera matrix shape: {np.array(camera_matrix).shape}")
    print(f"Distortion coefficients: {distortion_coeffs}")
    
    # Load image
    print(f"\nLoading image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    print(f"Image shape: {image.shape}")
    
    # Undistort image
    print("\nUndistorting image...")
    if model.lower() == 'fisheye':
        undistorted_image = undistort_fisheye_full_view(image, distortion_coeffs, camera_matrix)
    else:
        # For standard (non-fisheye) model, convert to numpy array
        undistorted_image = undistort_image(image, distortion_coeffs, camera_matrix)
    
    # Generate output filename in the same folder as input
    input_path = Path(image_path)
    output_filename = f"{input_path.stem}_undistorted{input_path.suffix}"
    output_path = input_path.parent / output_filename
    
    # Save undistorted image
    print(f"Saving undistorted image to: {output_path}")
    cv2.imwrite(str(output_path), undistorted_image)
    
    print(f"\n✓ Successfully saved undistorted image!")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Undistort an image using calibration data from JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i image.jpg -j calibration.json
  %(prog)s --image /path/to/image.png --json /path/to/intrinsic_camera_2.json
        """
    )
    
    parser.add_argument(
        '-i', '--image',
        type=str,
        required=True,
        help='Path to the input image to undistort'
    )
    
    parser.add_argument(
        '-j', '--json',
        type=str,
        required=True,
        help='Path to the calibration JSON file containing camera matrix and distortion coefficients'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}", file=sys.stderr)
        sys.exit(1)
    
    try:
        output_path = undistort_and_save(args.image, args.json)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
