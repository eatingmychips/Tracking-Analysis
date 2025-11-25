# app/tabs/tab1.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QHBoxLayout, QLineEdit, QGroupBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from tracking import core_tracking
import os 
from datetime import datetime 
import pandas as pd 
import numpy as np 

class MainWorker(QThread):
    # Define signals here for updating the GUI from the worker thread if needed
    pose_data_updated = pyqtSignal(list)
    finished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Accept references to parameters, or pass them as arguments
        # self.params = params

    def run(self):
        # Put main loop here – move video grabbing, serial reading, and pose calculation here
        # Call self.pose_data_updated.emit(pose_data_list) as data is produced, if you want GUI to update live
        pass

class Project1Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pose_data_list = []
        self.selected_directory = None
        self.init_ui()
        self.worker = None  # Thread to run main loop

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
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = os.path.join(self.selected_directory, f"pose_data_{timestamp}.csv")
            pose_data_list = [[10, 10, 10], [10, 10, 10], [10, 10, 10]]
            # Save pose data to CSV
            df = pd.DataFrame(pose_data_list, columns=['time', 'pose', 'arduino_data'])
            df.to_csv(output_filename, index=False)
            
            print(f"Recording stopped. Data saved to {output_filename}")
            pose_data_list = []  # Reset list after saving
            self.recording = False
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #5677fc; color: white; border-radius: 4px; padding: 4px 12px;")
        else: 
            self.recording = True 
            self.record_btn.setText("Stop Recording")
            self.record_btn.setStyleSheet("background-color: red; color: white;")
        pass

    # Add UI logic for updating widgets based on signals from the worker

# In your main window/tab widget setup:
# from app.tabs.tab1 import Project1Tab
# tab_widget.addTab(Project1Tab(), "Project 1 Recording")
