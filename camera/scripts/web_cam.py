import cv2
import os
import datetime
import time
import argparse
import av
import numpy as np

# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'video_codec;h264_nvenc'

def find_cameras():
    index = 0
    available_cameras = []
    print("Scanning for USB cameras...")
    # Scan first 10 indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Found camera at index: {i}")
            cap.release()
    return available_cameras

def record_video(cam_index=None, acceleration_factor=1.0):
    available = find_cameras()
    if cam_index is None:
        if not available:
            print("No USB cameras found.")
            return
        cam_index = available[0]
        
    print(f"Available cameras to use: {available}")
    print(f"Using camera at index: {cam_index}")
    print(f"Acceleration factor: {acceleration_factor}")

    # Setup directory
    # base_dir = os.path.join(os.getcwd(), 'captures', 'videos')
    base_dir = os.path.join(os.getcwd(), 'captures')
    session_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(base_dir, session_time)
    
    # Ensure directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {cam_index}.")
        return

    # Video settings
    fps = 20.0 * acceleration_factor
    container = None
    stream = None
    recording = False

    print("Controls:")
    print("  'a' - Start recording")
    print("  's' - Save and stop recording")
    print("  'q' - Quit script")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Show the frame
        cv2.imshow('Webcam Recording', frame)

        if recording and container is not None:
            # Convert BGR to RGB for av
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
            for packet in stream.encode(av_frame):
                container.mux(packet)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            if not recording:
                timestamp = int(time.time())
                filename = os.path.join(save_dir, f"{timestamp}.mp4")
                
                # Get frame dimensions
                height, width, layers = frame.shape
                
                container = av.open(filename, mode='w')
                stream = container.add_stream('libx264', rate=int(fps))
                stream.width = width
                stream.height = height
                stream.pix_fmt = 'yuv420p' # Standard for compatibility
                stream.options = {'preset': 'ultrafast', 'crf': '23'}
                
                recording = True
                print(f"Started recording: {filename}")
            else:
                print("Already recording.")

        elif key == ord('s'):
            if recording:
                recording = False
                # Flush the stream
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                container = None
                stream = None
                print("Video saved.")
            else:
                print("Not currently recording.")

        elif key == ord('q'):
            if recording:
                # Flush and close
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam Recording Script")
    parser.add_argument("-i", "--index", type=int, help="Camera index to use")
    parser.add_argument("-l", "--list", action="store_true", help="List available cameras and exit")
    parser.add_argument("-f", "--factor", type=float, default=1.0, help="Acceleration factor for the saved video")
    
    args = parser.parse_args()
    
    if args.list:
        find_cameras()
    else:
        record_video(cam_index=args.index, acceleration_factor=args.factor)
