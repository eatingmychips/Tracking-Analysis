# app/tabs/tab1.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QHBoxLayout, QLineEdit, QGroupBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
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
        self.selected_directory = None
        self.init_ui()
        self.worker = None  # Thread to run main loop
        self.recording = False

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

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
        main_layout.addWidget(port_group)

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
        main_layout.addWidget(dir_group)

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
        main_layout.addWidget(freq_group)

        # Record Button
        self.recording = False
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumWidth(120)
        self.record_btn.clicked.connect(self.toggle_recording)
        main_layout.addWidget(self.record_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Status Label
        self.result_label = QLabel("Status: Idle")
        self.result_label.setStyleSheet("QLabel { color: #333; }")
        main_layout.addWidget(self.result_label, alignment=Qt.AlignmentFlag.AlignLeft)

        main_layout.addStretch()
        self.setLayout(main_layout)

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
            self.selected_directory = directory
            self.dir_label.setText(directory)
        else: 
            self.dir_label.setText("(No folder Selected)")


    def toggle_recording(self):
        if self.recording:
            # Stop everything
            if self.worker:
                self.worker.stop()
                self.worker.wait()
            self.recording = False
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #5677fc; color: white;")
        else:
            # Build hardware objects
            config_file = r"L:\biorobotics\data\Vertical&InvertedClimbing\CameraFiles\VerticalClimbing.pfs"
            camera = core_tracking.PylonCamera(config_file)
            com_port = self.com_port_combo.currentText()
            serial_obj = serial.Serial(com_port, 115200, timeout=0.1)
            controller = core_tracking.JoystickController(self.freq_input.text, serial_obj)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector_params = (aruco_dict, parameters)

            self.pose_data_list = []
            session = core_tracking.TrackingSession(camera, controller, detector_params, self.pose_data_list)

            # Run in thread
            self.worker = MainWorker(session)
            self.worker.start()
            self.recording = True
            self.record_btn.setText("Stop Recording")
            self.record_btn.setStyleSheet("background-color: red; color: white;")

