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
import pygame 

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
    def __init__(self, frequency_var_getter, serial_obj):
        pygame.init()
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        self.joysticks = joysticks
        self.frequency_var_getter = frequency_var_getter
        self.serial_obj = serial_obj
        if joysticks:
            self.previous_button_states = [False] * joysticks[0].get_numbuttons()

    def process_input(self):
        data = ""
        dur = 500
        for joystick in self.joysticks:
            for button in range(joystick.get_numbuttons()):
                current_button_state = joystick.get_button(button)
                if current_button_state and not self.previous_button_states[button]:
                    if button == 0: # A Button
                        print("We have pressed button: 'A', stimulating both elytra")
                        side = "Both"
                        freq = int(self.frequency_var_getter())
                        message = get_command(dur, freq, side) + '\n'
                        self.serial_obj.write(message.encode('utf-8'))
                        data = (f"Both, {freq}")

                    elif button == 3: # Y Button
                        print("We have pressed button: 'Y', stimulating right antenna")
                        side = "Right"
                        freq = int(self.frequency_var_getter())
                        message = get_command(dur, freq, side) + '\n'
                        self.serial_obj.write(message.encode('utf-8'))
                        data = (f"Right, {freq}")

                    elif button == 2:  # X Button
                        print("We have pressed button: 'X', stimulating left antenna")
                        side = "Left"
                        freq = int(self.frequency_var_getter())
                        message = get_command(dur, freq, side) + '\n'
                        self.serial_obj.write(message.encode('utf-8'))
                        data = (f"Left, {freq}")
                    
                    elif button == 7: 
                        print("Start button pressed: Random Frequency")
                        # rand_freq()
                self.previous_button_states[button] = current_button_state


        return data

class TrackingSession:
    def __init__(self, camera, controller, detector_params, pose_data_list, recording_getter, directory_getter, show_cam_getter, frame_callback):
        self.camera = camera
        self.controller = controller
        self.detector = cv2.aruco.ArucoDetector(*detector_params)
        self.should_run = True
        self.pose_data_list = pose_data_list 
        self.directory_getter = directory_getter
        self.show_cam_getter = show_cam_getter
        self.frame_callback = frame_callback
        self.recording_getter = recording_getter 


    def run(self):
        self.camera.open()
        self.camera.start()
        try:
            while self.should_run:
                img = self.camera.get_frame()
                if img is not None:
                    data = self.controller.process_input()
                    corners, ids, rejected = self.detector.detectMarkers(img)
                    insect_pose = [None, None, None]
                    if ids is not None and 1 in ids:
                        idx = list(ids.flatten()).index(1)
                        marker_corners = corners[idx][0]
                        center = marker_corners.mean(axis=0)
                        dx, dy = marker_corners[1] - marker_corners[0]
                        angle = np.arctan2(dy, dx)
                        insect_pose = [center[0], center[1], angle]
                    
                    if self.recording_getter():
                        self.pose_data_list.append((time.time(), insect_pose, data))

                    if self.show_cam_getter and self.show_cam_getter(): 
                        self.frame_callback(img)
                        
        finally:
            self.camera.stop()

    def stop(self, stop_loop: bool = True):
        if stop_loop:
            self.should_run = False

        if len(self.pose_data_list) >= 1:
            df = pd.DataFrame(self.pose_data_list,
                              columns=['time', 'pose', 'arduino_data'])
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = os.path.join(
                self.directory_getter(),
                f"pose_data_{timestamp}.csv"
            )
            df.to_csv(output_filename, index=False)
            print(f"Data saved to {self.directory_getter()}")