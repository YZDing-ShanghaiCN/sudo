import depthai as dai
import os
# os.environ["DEPTHAI_LEVEL"] = "debug"

#import cv2
#import numpy as np
import argparse
import collections
import time
from itertools import cycle
from pathlib import Path
import sys
import signal
import math
#from stress_test import stress_test, YOLO_LABELS, create_yolo
#parser = argparse.ArgumentParser(add_help=False)
#args = parser.parse_args()
#success, device = dai.Device.getDeviceByMxId(args.device)
#print(args.device.getMxId())
device_infos = dai.Device.getAllAvailableDevices()
for device in device_infos:
    print(f"Device ID: {device.deviceId}")
    print(f"Name: {device.name}")
    print(f"Platform: {device.platform}")
    print(f"State: {device.state}")
    print(f"Protocol: {device.protocol}")
    print("---")
