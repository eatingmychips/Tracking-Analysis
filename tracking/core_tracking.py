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


class PylonCamera:
    def __init__(self, config_file):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.config_file = config_file

    def open(self):
        self.camera.Open()
        pylon.FeaturePersistence.Load(self.config_file, self.camera.GetNodeMap())

    def start(self):
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_frame(self):
        if self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                img = image.GetArray()
                return img
        return None

    def stop(self):
        self.camera.StopGrabbing()
        self.camera.Close()

class JoystickController:
    def __init__(self, joysticks, frequency_var, serial_obj):
        self.joysticks = joysticks
        self.frequency_var = frequency_var
        self.serial_obj = serial_obj
        if joysticks:
            self.previous_button_states = [False] * joysticks[0].get_numbuttons()

    def process_input(self):
        data = ""
        for joystick in self.joysticks:
            for button in range(joystick.get_numbuttons()):
                current_button_state = joystick.get_button(button)
                if current_button_state and not self.previous_button_states[button]:
                    # handle button logic as before
                    pass
                self.previous_button_states[button] = current_button_state
        return data

class TrackingSession:
    def __init__(self, camera, controller, detector_params):
        self.camera = camera
        self.controller = controller
        self.detector = cv2.aruco.ArucoDetector(*detector_params)
        self.recording = False
        self.pose_data_list = []

    def run(self):
        self.camera.open()
        self.camera.start()
        try:
            while True:
                img = self.camera.get_frame()
                if img is not None:
                    # process frame, aruco, etc.
                    pass
        finally:
            self.camera.stop()

    def stop(self): 
        self.camera.stop()