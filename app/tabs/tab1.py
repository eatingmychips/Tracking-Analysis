# app/tabs/tab1.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QHBoxLayout, QLineEdit, QGroupBox, QCheckBox, QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from tracking import core_tracking
import os 
from datetime import datetime 
import pandas as pd 
import numpy as np 
import serial
import cv2 

class MainWorker(QThread):
    pose_data_updated = pyqtSignal(list)
    finished = pyqtSignal()

    def __init__(self, session):
        super().__init__()
        self.session = session
        self._running = True

    def run(self):
        # Run session in this thread; implement a stop condition inside your session
        self.session.run()
        self.finished.emit()

    def stop(self):
        self.session.stop()  # Implement stop logic in TrackingSession
        self._running = False

class Project1Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pose_data_list = []
        self.init_ui()
        self.worker = None  # Thread to run main loop
        self.recording = False
        self.serial_obj = None
        self.camera = None

    def init_ui(self):
        # Root: horizontal split
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(15, 15, 15, 15)
        root_layout.setSpacing(12)

        # LEFT COLUMN (your existing controls)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        # Serial Port Group
        port_group = QGroupBox("Serial Port Settings")
        port_layout = QHBoxLayout()
        port_label = QLabel("Select COM Port:")
        port_label.setMinimumWidth(110)
        self.com_port_combo = QComboBox()
        self.com_port_combo.addItems(core_tracking.get_com_ports())
        self.com_port_combo.setMinimumWidth(120)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.com_port_combo)
        port_layout.addStretch()
        port_group.setLayout(port_layout)
        left_layout.addWidget(port_group)

        # Directory Selection
        dir_group = QGroupBox("Save Location")
        dir_layout = QHBoxLayout()
        self.dir_btn = QPushButton("Choose Directory")
        self.dir_btn.setMinimumWidth(150)
        self.dir_btn.clicked.connect(self.choose_directory)
        dir_layout.addWidget(self.dir_btn)
        self.dir_label = QLabel("(No Folder Selected)")
        dir_layout.addWidget(self.dir_label)
        dir_layout.addStretch()
        dir_group.setLayout(dir_layout)
        left_layout.addWidget(dir_group)

        # Frequency Input Group
        freq_group = QGroupBox("Recording Settings")
        freq_layout = QHBoxLayout()
        freq_label = QLabel("Frequency (Hz):")
        freq_label.setMinimumWidth(110)
        self.freq_input = QLineEdit("10")
        self.freq_input.setMinimumWidth(80)
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)
        freq_layout.addStretch()
        freq_group.setLayout(freq_layout)
        left_layout.addWidget(freq_group)

        # Record Button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumWidth(120)
        self.record_btn.clicked.connect(self.toggle_recording)
        left_layout.addWidget(self.record_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        left_layout.addStretch()  # push everything up
        # END LEFT COLUMN

        # RIGHT COLUMN (video + checkbox)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.camera_label)

        self.show_cam_checkbox = QCheckBox("Show camera stream")
        self.show_cam_checkbox.setChecked(False)
        self.show_cam_checkbox.toggled.connect(self.on_toggle_camera_view)
        right_layout.addWidget(self.show_cam_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        right_layout.addStretch()

        # Add left and right columns to root layout
        root_layout.addLayout(left_layout, stretch=0)   # content-determined width
        root_layout.addLayout(right_layout, stretch=1)  # expands to fill remaining space

        self.setLayout(root_layout)

        # Optional: Minimal stylesheet for a modern look
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #aaa; border-radius: 5px; margin-top: 10px;}
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px 0 5px; }
            QPushButton { background-color: #5677fc; color: white; border-radius: 4px; padding: 4px 12px;}
            QLineEdit, QComboBox { background-color: #f6f6f6; border-radius: 3px; }
        """)

    def choose_directory(self):
        dialog = QFileDialog(self)
        directory = dialog.getExistingDirectory()
        if directory:
            self.dir_label.setText(directory)
        else: 
            self.dir_label.setText("(No folder Selected)")


    def toggle_recording(self):
        if self.recording:
            self.pose_data_list = []
            self.recording = False
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #5677fc; color: white;")
            
            # Stop everything
            if self.worker and not self.show_cam_checkbox.isChecked:
                self.worker.stop()
                self.worker.wait()
                

            else: 
                self.worker.stop()
            
            if self.serial_obj is not None: 
                if self.serial_obj.is_open: 
                    self.serial_obj.close()
                self.serial_obj = None 

        elif self.worker is None:
            print(self.dir_label.text)
            if self.dir_label.text() == '(No Folder Selected)':
                QMessageBox.information(
                    self,                             # parent (e.g. your tab)
                    "Choose a folder",                # title
                    "Please choose a folder first.",  # text
                    QMessageBox.StandardButton.Ok     # buttons
                )
            else: 
                # Build hardware objects
                config_file = r"L:\biorobotics\data\Vertical&InvertedClimbing\CameraFiles\VerticalClimbing.pfs"

                self.camera = core_tracking.PylonCamera(config_file)
                com_port = self.com_port_combo.currentText()
                self.serial_obj = serial.Serial(com_port, 115200, timeout=0.1)
                controller = core_tracking.JoystickController(self.freq_input.text, self.serial_obj)
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters()
                detector_params = (aruco_dict, parameters)

                self.pose_data_list = []
                self.recording = True
                session = core_tracking.TrackingSession(self.camera, controller, detector_params, 
                                                        self.pose_data_list, recording_getter=self.get_recording_state, 
                                                        directory_getter=self.dir_label.text, 
                                                        show_cam_getter=self.show_cam_checkbox.isChecked,
                                                        frame_callback=self.update_camera_frame)

                # Run in thread
                self.worker = MainWorker(session)
                self.worker.start()
                
                self.record_btn.setText("Stop Recording")
                self.record_btn.setStyleSheet("background-color: red; color: white;")

        elif self.worker is not None: 
            print(self.dir_label.text)
            if self.dir_label.text() == '(No Folder Selected)': 
                QMessageBox.information(
                    self,                             # parent (e.g. your tab)
                    "Choose a folder",                # title
                    "Please choose a folder first.",  # text
                    QMessageBox.StandardButton.Ok     # buttons
                )
            else: 
                self.pose_data_list = []
                self.recording = True
                self.record_btn.setText("Stop Recording")
                self.record_btn.setStyleSheet("background-color: red; color: white;")

    def get_recording_state(self): 
        return self.recording
        

    def on_toggle_camera_view(self, checked: bool):
        if checked:
            if self.recording: 
                self.camera_label.show()
            if not self.recording: 
                config_file = r"L:\biorobotics\data\Vertical&InvertedClimbing\CameraFiles\VerticalClimbing.pfs"
                self.camera = core_tracking.PylonCamera(config_file)
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters()
                detector_params = (aruco_dict, parameters)
                controller = core_tracking.JoystickController(self.freq_input.text, self.serial_obj)
                session = core_tracking.TrackingSession(self.camera, controller, detector_params, 
                                                    self.pose_data_list, recording_getter=self.get_recording_state, 
                                                    directory_getter=self.dir_label.text, 
                                                    show_cam_getter=self.show_cam_checkbox.isChecked,
                                                    frame_callback=self.update_camera_frame)
                self.worker = MainWorker(session)
                self.worker.start()
                self.camera_label.show()

        else:
            if self.recording: 
                self.camera_label.hide()
            else: 
                if self.worker:
                    self.worker.stop()
                    self.worker.wait()
                    self.camera_label.hide()

    
    def update_camera_frame(self, img):
        # img: OpenCV BGR numpy array
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(pix)
