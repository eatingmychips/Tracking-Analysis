# app/tabs/tab1.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QHBoxLayout, QLineEdit, QGroupBox, QCheckBox, QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QIcon
from tracking import core_tracking
import os 
from datetime import datetime 
import pandas as pd 
import numpy as np 
import serial
import random
import pypylon
from pypylon import pylon
import cv2 

class MainWorker(QThread):
    finished = pyqtSignal()
     
    def __init__(self, session):
        super().__init__()
        self.session = session
        self._running = True

    def run(self):
        # Run session in this thread; implement a stop condition inside your session
        self.session.run()
        self.finished.emit()

    def stop(self, stop_loop: bool = True, save: bool = True):
        self.session.stop(stop_loop, save)  # Implement stop logic in TrackingSession
        self._running = False

class Project1Tab(QWidget):
    recording_changed = pyqtSignal(bool)
    show_cam_changed = pyqtSignal(bool)
    save_video_changed = pyqtSignal(bool)
    tracking_changed = pyqtSignal(bool)
    frequency_changed = pyqtSignal(int)
    directory_changed = pyqtSignal(str)
    filename_changed = pyqtSignal(str)

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
        nameing_group = QGroupBox("File Naming")
        naming_layout = QHBoxLayout()
        naming_label = QLabel("Filename:")
        naming_label.setMinimumWidth(40)
        self.naming_input = QLineEdit("data")
        self.naming_input.setMinimumWidth(150)


        # Frequency Input Group
        freq_group = QGroupBox("Frequency Settings")
        freq_layout = QHBoxLayout()
        freq_label = QLabel("Freq (Hz):")
        freq_label.setMinimumWidth(40)
        self.freq_input = QLineEdit("10")
        self.freq_input.setMinimumWidth(30)

        self.up_btn = QPushButton()
        self.up_btn.setIcon(QIcon("resources/uparrow.png"))
        self.up_btn.setFixedSize(40, 40)
        self.up_btn.setIconSize(QSize(35, 35))
        self.up_btn.clicked.connect(lambda: self.up_down_freq(True))

        self.down_btn = QPushButton()
        self.down_btn.setIcon(QIcon("resources/down_arrow.png"))
        self.down_btn.setFixedSize(40, 40)
        self.down_btn.setIconSize(QSize(35, 35))
        self.down_btn.clicked.connect(lambda: self.up_down_freq(False))

        self.rand_btn = QPushButton()  
        self.rand_btn.setIcon(QIcon("resources/dice.jpg"))
        self.rand_btn.setFixedSize(50, 50)
        self.rand_btn.setIconSize(QSize(46, 46))
        self.rand_btn.clicked.connect(self.rand_freq)  # your function

        # Remove blue background/border
        self.rand_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 0;
            }
            QPushButton:pressed {
                background-color: transparent;
            }
        """)

        self.up_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 0;
            }
            QPushButton:pressed {
                background-color: transparent;
            }
        """)

        self.down_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 0;
            }
            QPushButton:pressed {
                background-color: transparent;
            }
        """)
        
        naming_layout.addWidget(naming_label)
        naming_layout.addWidget(self.naming_input)
        naming_layout.addStretch()
        nameing_group.setLayout(naming_layout)
        left_layout.addWidget(nameing_group)

        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)
        freq_layout.addWidget(self.up_btn)
        freq_layout.addWidget(self.down_btn)
        freq_layout.addWidget(self.rand_btn)
        freq_layout.addStretch()
        freq_group.setLayout(freq_layout)
        left_layout.addWidget(freq_group)

        self.save_video_checkbox = QCheckBox("Save Video Enabled")
        self.save_video_checkbox.setChecked(False)
        self.save_video_checkbox.clicked.connect(self.on_toggle_save_video)
        left_layout.addWidget(self.save_video_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        self.enable_tracking = QCheckBox("Enable Live Tracking")
        self.enable_tracking.setChecked(True)
        self.enable_tracking.clicked.connect(self.on_toggle_tracking)
        left_layout.addWidget(self.enable_tracking, alignment=Qt.AlignmentFlag.AlignLeft)

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
        self.show_cam_checkbox.clicked.connect(self.on_toggle_camera_view)
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
            self.directory_changed.emit(directory)
            
        else: 
            self.dir_label.setText("(No folder Selected)")

    def start_session(self):
        try:
            config_file = r"L:\biorobotics\data\Vertical&InvertedClimbing\CameraFiles\ARUCO1200.pfs"
            self.camera = core_tracking.PylonCamera(config_file)
            com_port = self.com_port_combo.currentText()
            self.serial_obj = serial.Serial(com_port, 115200, timeout=0.1)
            controller = core_tracking.JoystickController(self.serial_obj)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector_params = (aruco_dict, parameters)

            self.pose_data_list = []
            self.session = core_tracking.TrackingSession(
                self.camera,
                controller,
                detector_params,
                self.pose_data_list
            )

            self.frequency_changed.connect(controller.set_frequency)
            self.frequency_changed.emit(int(self.freq_input.text()))

            self.save_video_changed.connect(self.session.set_save_video)
            self.save_video_changed.emit(self.save_video_checkbox.isChecked())

            self.tracking_changed.connect(self.session.set_save_tracking)
            self.tracking_changed.emit(self.enable_tracking.isChecked())

            self.recording_changed.connect(self.session.set_recording)

            self.show_cam_changed.connect(self.session.set_show_cam)
            self.show_cam_changed.emit(self.show_cam_checkbox.isChecked())

            self.filename_changed.connect(self.session.set_filename)
            self.filename_changed.emit(self.naming_input.text())

            self.directory_changed.connect(self.session.set_directory)
            self.directory_changed.emit(self.dir_label.text())

            self.worker = MainWorker(self.session)
            self.session.frame_ready.connect(self.update_camera_frame)
            self.worker.start()

        except Exception as e: 
            print("Error in start_session:", repr(e))
            QMessageBox.critical(self, "Error", f"Error in start_session:\n{e}")



    def toggle_recording(self):
        # STOP RECORDING
        if self.recording:
            self.recording_changed.emit(False)
            self.recording = False
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet(
                "background-color: #5677fc; color: white;"
            )
            self.enable_tracking.setEnabled(True)
            self.save_video_checkbox.setEnabled(True)
            if self.worker:
                if self.show_cam_checkbox.isChecked():
                    # Preview ON: save data but keep camera running
                    self.worker.stop(stop_loop=False, save = True)
                else:
                    # Preview OFF: save data and stop session
                    self.worker.stop(stop_loop=True, save = True)
                    self.worker.wait()
                    self.worker = None
                    if self.serial_obj is not None and self.serial_obj.is_open:
                        self.serial_obj.close()
                    self.serial_obj = None
            return

        # START RECORDING
        if self.dir_label.text() == '(No Folder Selected)':
            QMessageBox.information(
                self,
                "Choose a folder",
                "Please choose a folder first.",
                QMessageBox.StandardButton.Ok
            )
            return

        if self.worker is None:
            self.start_session()
        self.enable_tracking.setEnabled(False)
        self.save_video_checkbox.setEnabled(False)
        self.pose_data_list = []
        self.recording = True
        self.recording_changed.emit(True)
        self.record_btn.setText("Stop Recording")
        self.record_btn.setStyleSheet("background-color: red; color: white;")

    def get_recording_state(self): 
        return self.recording
    
    def rand_freq(self): 
        possible_freqs = [10, 20, 30, 40, 50]
        rand_freq = str(random.choice(possible_freqs))
        self.freq_input.setText(rand_freq)

    def up_down_freq(self, up: bool): 
        freq = int(self.freq_input.text())
        if up: 
            freq += 10
            if freq > 50: 
                freq = 10
        else: 
            freq -= 10
            if freq < 10: 
                freq = 50
        
        self.freq_input.setText(str(freq))
        
    def on_toggle_camera_view(self, checked: bool):
        if checked:
            self.camera_label.show()
            # If no session yet, start a preview-only session
            if self.worker is None:
                self.start_session()
                self.show_cam_changed.emit(checked)

        else:
            self.show_cam_changed.emit(checked)
            self.camera_label.clear()
            # If not recording, and user turns off preview, stop session entirely
            if not self.recording and self.worker:
                self.worker.stop(stop_loop=True, save = False)
                self.worker.wait()
                self.worker = None
                if self.serial_obj is not None and self.serial_obj.is_open:
                    self.serial_obj.close()
                self.serial_obj = None

    def on_toggle_save_video(self, checked: bool): 
        self.save_video_changed.emit(checked)

    def on_toggle_tracking(self, checked: bool): 
        self.tracking_changed.emit(checked)
    
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
