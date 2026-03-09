#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import argparse
import os
from datetime import datetime


def createPipeline(pipeline, camera_socket):
    """
    Create a pipeline with specified camera socket
    
    Args:
        pipeline: DepthAI pipeline object
        camera_socket: Camera socket to use (e.g., CAM_A, CAM_B, CAM_C)
    
    Returns:
        tuple: (pipeline, output queue)
    """
    camRgb = pipeline.create(dai.node.Camera).build(camera_socket)
    output = camRgb.requestOutput((1280, 800), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.CROP, 20).createOutputQueue()
    return pipeline, output


def get_camera_socket(camera_id):
    """
    Convert camera ID string to DepthAI camera socket
    
    Args:
        camera_id: Camera identifier (e.g., 'A', 'B', 'C', 'CAM_A', 'CAM_B', 'CAM_C')
    
    Returns:
        dai.CameraBoardSocket: Camera socket enum
    """
    camera_id = camera_id.upper().replace('CAM_', '')
    
    camera_map = {
        'A': dai.CameraBoardSocket.CAM_A,
        'B': dai.CameraBoardSocket.CAM_B,
        'C': dai.CameraBoardSocket.CAM_C,
        'D': dai.CameraBoardSocket.CAM_D,
        'E': dai.CameraBoardSocket.CAM_E,
        'F': dai.CameraBoardSocket.CAM_F,
    }
    
    if camera_id not in camera_map:
        raise ValueError(f"Invalid camera ID: {camera_id}. Valid options: A, B, C, D, E, F")
    
    return camera_map[camera_id]


def main():
    parser = argparse.ArgumentParser(description='Capture video from DepthAI camera devices')
    parser.add_argument('--device-id', '-d', type=str, default=None,
                        help='Specific device ID (MxId) to connect to. If not specified, connects to all available devices.')
    parser.add_argument('--camera-id', '-c', type=str, default='A',
                        help='Camera socket to use (A, B, C, D, E, F or CAM_A, CAM_B, etc.). Default: A')
    parser.add_argument('--list-devices', '-l', action='store_true',
                        help='List all available devices and exit')
    
    args = parser.parse_args()
    
    # Get all available devices
    deviceInfos = dai.Device.getAllAvailableDevices()
    
    if not deviceInfos:
        print("No DepthAI devices found!")
        return
    
    print("=== Found devices:")
    for i, deviceInfo in enumerate(deviceInfos):
        print(f"  [{i}] Device ID: {deviceInfo.deviceId}")
    
    if args.list_devices:
        return
    
    # Filter devices if specific device ID is requested
    if args.device_id:
        deviceInfos = [d for d in deviceInfos if args.device_id in d.deviceId]
        if not deviceInfos:
            print(f"Device with ID containing '{args.device_id}' not found!")
            return
        print(f"\n=== Using device: {deviceInfos[0].deviceId}")
    else:
        print("\n=== Using all available devices")
    
    # Get camera socket
    try:
        camera_socket = get_camera_socket(args.camera_id)
        print(f"=== Using camera: {args.camera_id}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    with contextlib.ExitStack() as stack:
        queues = []
        pipelines = []
        device_ids = []
        
        # Create session directory for saving images
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        session_dir = os.path.join("captures", session_name)
        os.makedirs(session_dir, exist_ok=True)
        print(f"\n=== Session directory created: {session_dir}")

        for deviceInfo in deviceInfos:
            device_info = dai.DeviceInfo(args.device_id)
            device = dai.Device(device_info)
            pipeline = stack.enter_context(dai.Pipeline(defaultDevice=device))


            print(f"\n=== Connected to {deviceInfo.deviceId}")
            mxId = device.getDeviceId()
            cameras = device.getConnectedCameras()
            usbSpeed = device.getUsbSpeed()
            eepromData = device.readCalibration2().getEepromData()
            
            print(f"   >>> Device ID: {mxId}")
            print(f"   >>> Num of cameras: {len(cameras)}")
            print(f"   >>> Connected cameras: {[str(cam) for cam in cameras]}")
            print(f"   >>> USB speed: {usbSpeed.name}")
            print(f"   >>> Use camera socket: {camera_socket.name}")

            if eepromData.boardName != "":
                print(f"   >>> Board name: {eepromData.boardName}")
            if eepromData.productName != "":
                print(f"   >>> Product name: {eepromData.productName}")
            
            # Check if requested camera is available
            if camera_socket not in cameras:
                print(f"   >>> WARNING: Camera {args.camera_id} not available on this device!")
                continue
            
            pipeline, output = createPipeline(pipeline, camera_socket)
            pipeline.start()
            pipelines.append(pipeline)
            queues.append(output)
            device_ids.append(mxId)

        if not queues:
            print("\nNo devices with the specified camera available!")
            return
        
        print("\n=== Starting video capture. Press 'q' to quit, 's' to save image.")
        
        # Store current frames for saving
        current_frames = {}
        
        while True:
            for i, stream in enumerate(queues):
                videoIn = stream.get()
                assert isinstance(videoIn, dai.ImgFrame)
                frame = videoIn.getCvFrame()
                device_id = device_ids[i]
                window_name = f"Device_{device_id[:8]}_CAM_{args.camera_id}"
                current_frames[device_id] = frame
                cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save all current frames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
                for device_id, frame in current_frames.items():
                    filename = f"{device_id}_CAM_{args.camera_id}_{timestamp}.png"
                    filepath = os.path.join(session_dir, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"   >>> Saved: {filename}")
                print(f"=== Saved {len(current_frames)} image(s) at {timestamp}")
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
