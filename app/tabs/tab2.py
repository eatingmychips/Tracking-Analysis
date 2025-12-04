# app/tabs/tab2.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox
from PyQt6.QtWidgets import QSizePolicy, QHBoxLayout, QLineEdit, QGroupBox, QCheckBox, QFileDialog, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from analysis import core_analysis
import time
from dataclasses import dataclass 


@dataclass
class AnalysisOptions: 
    antenna_time: bool = False 
    antenna_freq: bool = False 
    elytra_time: bool = False 
    elytra_freq: bool = False 


class AnalysisWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    results_ready = pyqtSignal(object)
    def __init__(self, directory, options: AnalysisOptions, parent=None): 
        super().__init__(parent)
        self.directory = directory
        self.options = options
        self.finished.connect()

    def run(self): 
        try: 
            results = core_analysis.run_analysis(directory = self.directory)
            self.results_ready.emit(results)
        except Exception as e: 
            self.error.emit(str(e))

        finally: 
            self.finished.emit()


class Project2Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.directory = None
        self.init_ui()
        self.worker = None  # Thread to run main loop

    def init_ui(self):
        # Root: horizontal split
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(15, 15, 15, 15)
        root_layout.setSpacing(12)

        # LEFT COLUMN (your existing controls)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        ### Directory Selection ###
        dir_group = QGroupBox("Select Folder")
        dir_layout = QHBoxLayout()
        self.dir_btn = QPushButton("Choose Directory")
        self.dir_btn.setMinimumWidth(150)
        self.dir_btn.clicked.connect(self.choose_directory)
        dir_layout.addWidget(self.dir_btn)
        self.dir_label = QLabel("(No Folder Selected)")
        self.dir_label.setFixedWidth(150)          # pick a width that fits your layout
        self.dir_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
        )
        self.dir_label.setWordWrap(False)          # single line, no height growth
        dir_layout.addWidget(self.dir_label)
        dir_layout.addStretch()
        dir_group.setLayout(dir_layout)
        left_layout.addWidget(dir_group)
        ### End Directory Selection ###
        
        ### Checbox Layout, here we have checkboxes to identify the type of plots to show ###
        checkbox_group = QGroupBox("Analysis to run")
        checkbox_layout = QVBoxLayout()
        self.ant_angle_box = QCheckBox("Time - Angle (Antenna)")
        self.ant_freq_box = QCheckBox("Frequency - Angle (Antenna)")
        self.ely_vel_box = QCheckBox("Time - Velocity (Both Elytra)")
        self.ely_freq_box = QCheckBox("Freq - Vel (Elytra)")
        
        checkbox_layout.addWidget(self.ant_angle_box, alignment=Qt.AlignmentFlag.AlignLeft)
        checkbox_layout.addWidget(self.ant_freq_box, alignment=Qt.AlignmentFlag.AlignLeft)
        checkbox_layout.addWidget(self.ely_vel_box, alignment=Qt.AlignmentFlag.AlignLeft)
        checkbox_layout.addWidget(self.ely_freq_box, alignment=Qt.AlignmentFlag.AlignLeft)

        checkbox_group.setLayout(checkbox_layout)
        ### End Checkbox ### 

        ### Run Analysis Button ###
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.setMinimumWidth(120)
        self.run_analysis_btn.clicked.connect(self.run_analysis_clicked)

        
        
        left_layout.addWidget(checkbox_group)     
        left_layout.addWidget(self.run_analysis_btn, alignment=Qt.AlignmentFlag.AlignLeft)   
        left_layout.addStretch()
        # END LEFT COLUMN 


        # RIGHT COLUMN (video + checkbox)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.camera_label)


        root_layout.addLayout(left_layout)
        root_layout.addLayout(right_layout)
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
        self.directory = dialog.getExistingDirectory()
        if self.directory:
            self.dir_label.setText(self.directory)
            
        else: 
            self.dir_label.setText("(No folder Selected)")  


    def run_analysis_clicked(self):
        self.run_analysis_btn.setText("Running Analysis")
        self.run_analysis_btn.setStyleSheet("background-color: red; color: white;")
        options = AnalysisOptions(
            antenna_time=self.ant_angle_box.isChecked(),
            antenna_trials=self.ant_freq_box.isChecked(),
            elytra_time=self.ely_vel_box.isChecked(),
            elytra_trials=self.ant_freq_box.isChecked(),
        )
        self.analysis_worker = AnalysisWorker(self.directory, options)
        self.analysis_worker.results_ready.connect(self.on_analysis_results)

        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.start()
    
        
    def reset_run_button(self):
        self.run_analysis_btn.setText("Run Analysis")
        self.run_analysis_btn.setStyleSheet("background-color: #5677fc; color: white;")


    def on_analysis_finished(self): 
        self.run_analysis_btn.setText("Run Analysis")
        self.run_analysis_btn.setStyleSheet("background-color: #5677fc; color: white;")
        self.run_analysis_btn.setEnabled(True)
        self.analysis_worker = None


    def on_analysis_results(self, results): 
        self.plot_results = results
        self.update_plot_view


    def antenna_time_plot(self, body_angles, frequencies): 