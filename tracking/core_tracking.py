# project1/core.py
import pypylon
from pypylon import pylon
import cv2
import sys
import numpy as np
import pandas as pd
import serial
import threading
import time
import os 
from datetime import datetime
import random
import pygame 
from dataclasses import dataclass 
from typing import Callable, Optional 
from PyQt6.QtCore import QObject, pyqtSignal


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

class JoystickController(QObject):
    random_button = pyqtSignal(bool)
    def __init__(self, serial_obj):
        super().__init__()
        pygame.init()
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        self.joysticks = joysticks
        self.serial_obj = serial_obj
        self.frequency = 10
        if joysticks:
            self.previous_button_states = [False] * joysticks[0].get_numbuttons()

    def set_frequency(self, value: int): 
        self.frequency = value

    def process_input(self):
        data = ""
        dur = 500
        for joystick in self.joysticks:
            for button in range(joystick.get_numbuttons()):
                current_button_state = joystick.get_button(button)
                if current_button_state and not self.previous_button_states[button]:
                    if button == 0:      # A
                        print("We have pressed button: 'A', stimulating both elytra")
                        side = "Both"
                    elif button == 3:    # Y
                        print("We have pressed button: 'Y', stimulating right antenna")
                        side = "Right"
                    elif button == 2:    # X
                        print("We have pressed button: 'X', stimulating left antenna")
                        side = "Left"
                    elif button == 7:
                        print("Start button pressed: Random Frequency")
                        side = None
                        self.random_button.emit(True)

                    if side is not None:
                        freq = int(self.frequency)
                        message = get_command(dur, freq, side) + '\n'
                        self.serial_obj.write(message.encode('utf-8'))
                        data = f"{side}, {freq}"


                self.previous_button_states[button] = current_button_state


        return data

class TrackingSession(QObject):
    frame_ready = pyqtSignal(object)  # img (numpy array)
    def __init__(self, camera, controller, detector_params, pose_data_list):
        super().__init__()
        self.camera = camera
        self.controller = controller
        self.detector = cv2.aruco.ArucoDetector(*detector_params)
        self.should_run = True
        self.pose_data_list = pose_data_list 
        self.video_writer = None

        self.recording = False
        self.show_cam = False
        self.save_video = False 
        self.save_tracking = True

        self.filename = ""
        self.directory = ""
        self.frequency = 10

        self.prev_center = None
        self.crop_radius = 200


    def run(self):
        self.video_writer = None
        self.camera.open()
        self.camera.start()
        try:
            while self.should_run:
                img = self.camera.get_frame()
                insect_pose = [None, None, None]
                data = self.controller.process_input()
                if img is not None:
                    if self.recording:    
                        if self.save_tracking:    
                            if self.prev_center is not None: 
                                cx, cy = self.prev_center
                                x0 = max(int(cx - self.crop_radius), 0)
                                y0 = max(int(cy - self.crop_radius), 0)
                                x1 = min(int(cx + self.crop_radius), img.shape[1])
                                y1 = min(int(cy + self.crop_radius), img.shape[0])
                                roi = img[y0:y1, x0:x1]
                                corners, ids, rejected = self.detector.detectMarkers(roi)
                            else: 
                                x0 = y0 = 0
                                corners, ids, rejected = self.detector.detectMarkers(img)

                            if ids is not None and 1 in ids:
                                idx = list(ids.flatten()).index(1)
                                marker_corners = corners[idx][0]

                                # offset ROI back to full image coords
                                marker_corners[:, 0] += x0
                                marker_corners[:, 1] += y0

                                center = marker_corners.mean(axis=0)
                                dx, dy = marker_corners[1] - marker_corners[0]
                                angle = np.arctan2(dy, dx)
                                insect_pose = [center[0], center[1], angle]

                                # update prev_center in full-res coords
                                self.prev_center = (center[0], center[1])
                            else: 
                                self.prev_center = None
                                
                        if self.save_video:
                            if self.video_writer is None:
                                h, w, _ = img.shape
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                video_filename = os.path.join(
                                    self.directory,
                                    f"{self.filename}_{timestamp}.avi"
                                )
                                self.video_writer = cv2.VideoWriter(
                                    video_filename, fourcc, 100, (w, h)
                                )
                            self.video_writer.write(img)
                            
                        self.pose_data_list.append((time.time(), insect_pose, data))
                        
                    if self.show_cam: 
                        self.frame_ready.emit(img)
                        
        finally:
            self.camera.stop()

    def stop(self, stop_loop: bool = True, save: bool = True):
        if stop_loop:
            self.should_run = False

        # Close video writer and reset
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if save: 
            if len(self.pose_data_list) >= 1:
                df = pd.DataFrame(self.pose_data_list,
                                columns=['time', 'pose', 'arduino_data'])
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = os.path.join(
                    self.directory,
                    f"{self.filename}_pose_{timestamp}.csv"
                )
                df.to_csv(output_filename, index=False)
                print(f"Data saved to {self.directory}")

            self.pose_data_list = []
        else: 
            return 
        
    
    def set_recording(self, value: bool): 
        self.recording = value

    def set_show_cam(self, value: bool): 
        self.show_cam = value
    
    def set_save_video(self, value: bool): 
        self.save_video = value

    def set_save_tracking(self, value: bool): 
        self.save_tracking = value

    def set_filename(self, value: str): 
        self.filename = value

    def set_directory(self, value: str): 
        self.directory = value
