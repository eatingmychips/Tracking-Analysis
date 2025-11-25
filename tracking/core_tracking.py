# project1/core.py
import cv2
from pypylon import pylon
import numpy as np
import pandas as pd
import serial
import threading
import time
import os 
from datetime import datetime
import random

# Example: logic for arduino/serial handling, video tracking, etc.

def get_com_ports():
    import serial.tools.list_ports
    return [port.device for port in serial.tools.list_ports.comports()]

def save_pose_data(selected_directory, pose_data_list):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = os.path.join(selected_directory, f"pose_data_{timestamp}.csv")
    df = pd.DataFrame(pose_data_list, columns=['time', 'pose', 'arduino_data'])
    df.to_csv(output_filename, index=False)
    return output_filename

def get_command(dur, freq, selection):
    dur_hex = hex(int(dur/10))[2:].zfill(2)
    freq_hex = hex(int(freq))[2:].zfill(2)
    prefix = "B1"
    cmds = {"Both": "E0", "Left": "A0", "Right": "B0"}
    return prefix + cmds[selection] + dur_hex + freq_hex

# Your other computational/IO handling functions go here
# Consider wrapping major procedures (e.g., video recording loop, aruco detection) in classes or functions
class ArucoDetection()