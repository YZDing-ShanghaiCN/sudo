#!/usr/bin/env python3

"""
Camera Capture Script - Simplified version of cam_test.py for capturing images

Usage:
    python cam_capture.py -d <device_id> -i <camera_id> [options]
    
    -d, --device: Device MX ID or camera group to connect to
    -i, --camera-id: Camera socket to use (rgb, left, right, cama, camb, camc, camd, came)
    
Examples:
    python cam_capture.py -d <device_mx_id> -i rgb
    python cam_capture.py -d <device_mx_id> -i left -n 10
"""

import depthai as dai
import cv2
import argparse
import time
import numpy as np
from pathlib import Path

# All available camera sockets
ALL_SOCKETS = ['rgb', 'left', 'right', 'cama', 'camb', 'camc', 'camd', 'came']

# Camera socket mapping
cam_socket_opts = {
    'rgb': dai.CameraBoardSocket.CAM_A,
    'left': dai.CameraBoardSocket.CAM_B,
    'right': dai.CameraBoardSocket.CAM_C,
    'cama': dai.CameraBoardSocket.CAM_A,
    'camb': dai.CameraBoardSocket.CAM_B,
    'camc': dai.CameraBoardSocket.CAM_C,
    'camd': dai.CameraBoardSocket.CAM_D,
    'came': dai.CameraBoardSocket.CAM_E,
}

mono_res_opts = {
    400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
    480: dai.MonoCameraProperties.SensorResolution.THE_480_P,
    720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
    800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
}

color_res_opts = {
    '720':  dai.ColorCameraProperties.SensorResolution.THE_720_P,
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '4k':   dai.ColorCameraProperties.SensorResolution.THE_4_K,
}


class FPSCounter:
    def __init__(self):
        self.frameTimes = []

    def tick(self):
        now = time.time()
        self.frameTimes.append(now)
        self.frameTimes = self.frameTimes[-100:]

    def getFps(self):
        if len(self.frameTimes) <= 1:
            return 0
        # Calculate the FPS
        return (len(self.frameTimes) - 1) / (self.frameTimes[-1] - self.frameTimes[0])

def parse_args():
    parser = argparse.ArgumentParser(
        description='Camera Capture Script - Capture images from DepthAI cameras',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('-d', '--device', type=str, default="",
                        help='Device MX ID or camera group to connect to')
    
    parser.add_argument('-i', '--camera-id', type=str, default="",
                        help='Camera socket ID to use (e.g., rgb, left, right, cama, etc.). If not specified, lists available cameras')
    
    parser.add_argument('-o', '--output-dir', type=str, default='./captures',
                        help='Output directory for captured images (default: ./captures)')
    
    parser.add_argument('-n', '--num-captures', type=int, default=1,
                        help='Number of images to capture (default: 1, 0 for continuous)')
    
    parser.add_argument('-f', '--fps', type=float, default=30,
                        help='Camera FPS (default: 30)')
    
    parser.add_argument('-mres', '--mono-resolution', type=int, default=800, 
                        choices=[480, 400, 720, 800],
                        help='Mono camera resolution height (default: 800)')
    
    parser.add_argument('-cres', '--color-resolution', default='1080',
                        choices=['720', '1080', '4k'],
                        help='Color camera resolution (default: 1080)')
    
    parser.add_argument('-p', '--preview', action='store_true',
                        help='Show preview window')
    
    parser.add_argument('--interval', type=float, default=0.0,
                        help='Interval between captures in seconds (default: 0, capture as fast as possible)')
    
    parser.add_argument('--prefix', type=str, default='capture',
                        help='Prefix for captured image filenames (default: capture)')
    
    parser.add_argument('--show-calib', action='store_true',
                        help='Show camera calibration data (intrinsics, distortion coefficients, FOV) and exit')
    
    parser.add_argument('--width', type=int, default=640,
                        help='Width of the captured image (default: 640)')
    
    parser.add_argument('--height', type=int, default=480,
                        help='Height of the captured image (default: 480)')
    
    parser.add_argument('--crop-mode', type=int, default=0,
                        choices=[0, 1, 2],
                        help='Crop mode for image capture: 0-CROP, 1-STRETCH, 2-LETTERBOX (default: 0)')
    
    return parser.parse_args()

def socket_to_socket_opt(socket: dai.CameraBoardSocket) -> str:
    return str(socket).split('.')[-1].replace("_", "").lower()

def connect_to_device(device_id):
    """Connect to a DepthAI device by MX ID or return any available device"""
    dai_device_args = []
    if device_id:
        device_info = dai.DeviceInfo(device_id)
        dai_device_args.append(device_info)
        print(f"Connecting to device: {device_id}")
    else:
        print("Connecting to any available device...")
    
    return dai.Device(*dai_device_args)

def show_calibration_data(device, connected_cameras):
    """Display camera calibration data including intrinsics and distortion coefficients"""
    print("\n" + "="*80)
    print("CAMERA CALIBRATION DATA")
    print("="*80)
    
    calibData = device.readCalibration()
    eeprom = calibData.getEepromData()
    print(f"\nDevice: {device.getMxId()}")
    print(f"Board: {eeprom.boardName}")
    print(f"Board Rev: {eeprom.boardRev}")
    
    for cam_feature in connected_cameras:
        socket = cam_feature.socket
        socket_name = socket_to_socket_opt(socket)
        
        print(f"\n{'-'*80}")
        print(f"Camera: {socket_name} ({socket.name}) - {cam_feature.sensorName}")
        print(f"{'-'*80}")
        
        # Get intrinsics
        try:
            M, width, height = calibData.getDefaultIntrinsics(socket)
            print(f"\nDefault Intrinsics ({width}x{height}):")
            print(f"  Focal Length: fx={M[0][0]:.2f}, fy={M[1][1]:.2f}")
            print(f"  Principal Point: cx={M[0][2]:.2f}, cy={M[1][2]:.2f}")
            print(f"  Intrinsic Matrix:\n{np.array(M)}")
            
            # Get distortion coefficients
            D = np.array(calibData.getDistortionCoefficients(socket))
            print(f"\nDistortion Coefficients:")
            distortion_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", 
                               "s1", "s2", "s3", "s4", "τx", "τy"]
            for name, value in zip(distortion_names, D):
                print(f"  {name:3s}: {value:>12.8f}")
            
            # Get FOV
            fov = calibData.getFov(socket)
            print(f"\nField of View: {fov:.2f}°")
            
        except Exception as e:
            print(f"  No calibration data available: {e}")
    
    # Show stereo pair info if available
    try:
        left_socket = eeprom.stereoRectificationData.leftCameraSocket
        right_socket = eeprom.stereoRectificationData.rightCameraSocket
        
        print(f"\n{'-'*80}")
        print(f"STEREO PAIR")
        print(f"{'-'*80}")
        print(f"Left Camera: {socket_to_socket_opt(left_socket)} ({left_socket.name})")
        print(f"Right Camera: {socket_to_socket_opt(right_socket)} ({right_socket.name})")
        
        # Get baseline
        lr_extrinsics = np.array(calibData.getCameraExtrinsics(left_socket, right_socket))
        baseline = np.linalg.norm(lr_extrinsics[:3, 3])
        print(f"Baseline: {baseline*10:.2f} cm")
        
        print(f"\nLeft->Right Extrinsics (transformation matrix):")
        print(lr_extrinsics)
        
    except Exception as e:
        print(f"\nNo stereo pair calibration available: {e}")
    
    print(f"\n{'='*80}\n")

def list_available_cameras(connected_cameras):
    """List all available cameras on the device"""
    print("\nAvailable cameras:")
    print(f"{'Socket ID':<12} {'Sensor':<20} {'Resolution':<15} {'Type':<10} {'Focus'}")
    print("-" * 75)
    for cam_feature in connected_cameras:
        socket_name = socket_to_socket_opt(cam_feature.socket)
        sensor = cam_feature.sensorName
        resolution = f"{cam_feature.width}x{cam_feature.height}"
        cam_type = 'color' if cam_feature.supportedTypes[0] == dai.CameraSensorType.COLOR else \
                  'mono' if cam_feature.supportedTypes[0] == dai.CameraSensorType.MONO else \
                  'tof' if cam_feature.supportedTypes[0] == dai.CameraSensorType.TOF else \
                  'thermal'
        focus = 'auto' if cam_feature.hasAutofocus else 'fixed'
        print(f"{socket_name:<12} {sensor:<20} {resolution:<15} {cam_type:<10} {focus}")
    
    print("\nUsage: python cam_capture.py -i <camera_id> [options]")
    print("Example: python cam_capture.py -i rgb -n 5")

def find_camera(camera_id, connected_cameras):
    """Find and validate the specified camera on the device"""
    target_socket = cam_socket_opts[camera_id]
    
    for cam_feature in connected_cameras:
        if cam_feature.socket == target_socket:
            is_color = cam_feature.supportedTypes[0] == dai.CameraSensorType.COLOR
            print(f"Found camera at socket {camera_id}")
            print(f"  Sensor: {cam_feature.sensorName}")
            print(f"  Resolution: {cam_feature.width}x{cam_feature.height}")
            print(f"  Type: {'color' if is_color else 'mono'}")
            return cam_feature, is_color
    
    # Camera not found
    print(f"Error: Camera not found at socket '{camera_id}'")
    print("Available cameras:")
    for cam_feature in connected_cameras:
        socket_name = socket_to_socket_opt(cam_feature.socket)
        print(f"  - {socket_name} ({cam_feature.sensorName})")
    return None, False

def create_pipeline(camera_id, is_color, args):
    """Create DepthAI pipeline for the specified camera using the new API"""
    pipeline = dai.Pipeline()

    # Initialize ImgFrameCapability
    cap = dai.ImgFrameCapability()
    cap.size.fixed((int(args.width), int(args.height)))
    cropArg = int(args.crop_mode)

    if cropArg == 0:
        cap.resizeMode = dai.ImgResizeMode.CROP
    elif cropArg == 1:
        cap.resizeMode = dai.ImgResizeMode.STRETCH
    elif cropArg == 2:
        cap.resizeMode = dai.ImgResizeMode.LETTERBOX
    else:
        raise ValueError("Invalid crop mode")

    cap.fps.fixed(float(args.fps))

    # Determine camera socket
    camArg = camera_id.upper()
    if camArg == "CAMA":
        socket = dai.CameraBoardSocket.CAM_A
    elif camArg == "CAMB":
        socket = dai.CameraBoardSocket.CAM_B
    elif camArg == "CAMC":
        socket = dai.CameraBoardSocket.CAM_C
    elif camArg == "CAMD":
        socket = dai.CameraBoardSocket.CAM_D
    else:
        raise ValueError("Invalid camera socket")

    # Create camera node
    cams = {}
    if socket not in cams:
        cams[socket] = pipeline.create(dai.node.Camera).build(socket)

    # Create output queue
    queues = []
    queues.append(cams[socket].requestOutput(cap, True).createOutputQueue())

    return pipeline, queues

def capture_images(device, camera_id, is_color, args):
    """Main capture loop for taking images from the camera"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create and start pipeline
    pipeline, queues = create_pipeline(camera_id, is_color, args)
    pipeline.start()
    FPSCounters = [FPSCounter() for _ in queues]
    # Get output queue
    # queue = device.getOutputQueue(name='capture', maxSize=4, blocking=False)
    
    # Capture loop
    capture_count = 0
    continuous = args.num_captures == 0
    last_capture_time = 0
    
    print(f"\nStarting capture... (Press 'q' to quit)")
    if continuous:
        print("Continuous mode - capturing indefinitely")
    else:
        print(f"Capturing {args.num_captures} image(s)")
    
    try:
        wFPSCounters = [FPSCounter() for _ in queues]
        while pipeline.isRunning():
            for index, queue in enumerate(queues):
                videoIn = queue.tryGet()
                if videoIn is not None:
                    FPSCounters[index].tick()
                    assert isinstance(videoIn, dai.ImgFrame)
                    print(
                        f"frame {videoIn.getWidth()}x{videoIn.getHeight()} | {videoIn.getSequenceNum()}: exposure={videoIn.getExposureTime()}us, timestamp: {videoIn.getTimestampDevice()}"
                    )
                    # Get BGR frame from NV12 encoded video frame to show with opencv
                    # Visualizing the frame on slower hosts might have overhead
                    cvFrame = videoIn.getCvFrame()
                    # Draw FPS
                    cv2.putText(cvFrame, f"{FPSCounters[index].getFps():.2f} FPS", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.imshow("video " + str(index), cvFrame)

            if cv2.waitKey(1) == ord("q"):
                break
        
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    print(f"\nCapture complete. Total images captured: {capture_count}")
    print(f"Images saved to: {output_dir}")
    
    if args.preview:
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    
    print(f"DepthAI version: {dai.__version__}")
    
    # Connect to device
    device = connect_to_device(args.device)
    if device is None:
        return
    
    with device:
        # Get camera features
        connected_cameras = device.getConnectedCameraFeatures()
        
        # Show calibration data if requested
        if args.show_calib:
            show_calibration_data(device, connected_cameras)
            return
        
        # If no camera ID specified, list available cameras and exit
        if not args.camera_id:
            list_available_cameras(connected_cameras)
            return
        
        # Validate camera ID
        if args.camera_id not in ALL_SOCKETS:
            print(f"Error: Invalid camera ID '{args.camera_id}'")
            print(f"Valid options: {', '.join(ALL_SOCKETS)}")
            return
        
        print(f"Camera ID: {args.camera_id}")
        
        # Find the specified camera
        cam_feature, is_color = find_camera(args.camera_id, connected_cameras)
        if cam_feature is None:
            return
        
        # Capture images
        capture_images(device, args.camera_id, is_color, args)

if __name__ == '__main__':
    main()
