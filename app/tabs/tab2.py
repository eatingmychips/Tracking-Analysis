# app/tabs/tab2.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox
from PyQt6.QtWidgets import QSizePolicy, QHBoxLayout, QLineEdit, QGroupBox, QCheckBox, QFileDialog, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from analysis import core_analysis
import time
from dataclasses import dataclass 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np 


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

        self.figure = Figure(figsize = (12, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

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
            antenna_freq=self.ant_freq_box.isChecked(),
            elytra_time=self.ely_vel_box.isChecked(),
            elytra_freq=self.ant_freq_box.isChecked(),
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
        data_dict = results["body_angles"]
        frequencies = [10, 20, 30, 40, 50]
        self.antenna_time_plot(data_dict, frequencies, "Angular Deviation (degrees)")

    def resample_1d_list(self, original_list, new_len):
        # Remove invalid (non-float) entries
        filtered = [x for x in original_list if isinstance(x, (float, int, np.float32, np.float64))]
        old_len = len(filtered)
        if old_len == 0:
            # Return a list of np.nan if no valid numbers remain
            return [np.nan] * new_len
        if old_len == 1:
            return [filtered] * new_len
        old_idx = np.linspace(0, 1, old_len)
        new_idx = np.linspace(0, 1, new_len)
        return np.interp(new_idx, old_idx, filtered).tolist()


    def antenna_time_plot(self, data_dict, frequencies, title):

        self.figure.clear()

        # create 2x3 grid of subplots on the existing figure
        axes = self.figure.subplots(2, 3)
        axes_flat = axes.flatten()

        def process_list(data):
            data = np.array(data, dtype=object)
            max_len = max(len(sublist) for sublist in data)

            resampled_data = np.array([
            self.resample_1d_list(sublist, max_len) for sublist in data
                ], dtype=float)
            means = np.nanmean(resampled_data, axis=0)
            stds = np.nanstd(resampled_data, axis=0)
            lower = means - stds     # One std dev below the mean
            upper = means + stds     # One std dev above the mean
            return max_len, means, lower, upper

        axes_flat = axes.flatten()

        for idx, freq in enumerate(frequencies):
            ax = axes_flat[idx]

            
            # Use .get() with default empty list if key not found
            list1 = data_dict.get(("Right", freq), [])
            list2 = data_dict.get(("Left", freq), [])


            if len(list1) < 1 and len(list2) < 1:
                # If no data at all for this frequency, just create empty plot
                ax.set_title(f'Freq: {freq} Hz (No Data)', fontsize=18)
                ax.set_xlim(0, 1.15)
                ax.set_ylabel(title, fontsize=16)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                continue
        
            # Only process and plot if data exists
            if len(list1) > 0:
                max_len1, medians1, lower_quartiles1, upper_quartiles1 = process_list(list1)
                x1 = np.linspace(0, 1.15, max_len1)
                mask1 = (x1 >= 0.15) & (x1 <= 0.65)
                ax.fill_between(x1, lower_quartiles1, upper_quartiles1, color='lightgrey', alpha=0.3)
                ax.plot(x1, medians1, color='black', linewidth=2)
                ax.fill_between(x1[mask1], lower_quartiles1[mask1], upper_quartiles1[mask1], color='lightcoral', alpha=0.3)
                ax.plot(x1[mask1], medians1[mask1], color='red', linewidth=2, label='Right Stimulation')

            if len(list2) > 0:
                max_len2, medians2, lower_quartiles2, upper_quartiles2 = process_list(list2)
                x2 = np.linspace(0, 1.15, max_len2)
                mask2 = (x2 >= 0.15) & (x2 <= 0.65)
                ax.fill_between(x2, lower_quartiles2, upper_quartiles2, color='lightgrey', alpha=0.3)
                ax.plot(x2, medians2, color='black', linewidth=2)
                ax.fill_between(x2[mask2], lower_quartiles2[mask2], upper_quartiles2[mask2], color='lightgreen', alpha=0.3)
                ax.plot(x2[mask2], medians2[mask2], color='green', linewidth=2, label='Left Stimulation')

                


            # Formatting subplot
            ax.set_title(f'Freq: {freq} Hz', fontsize=18)
            ax.set_xlim(0, 1.1)
            ax.set_ylim(-45, 45)
            ax.set_ylabel(title, fontsize=16)
            if freq == 10:
                ax.legend(fontsize=14)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if len(frequencies) < len(axes_flat):
            self.figure.delaxes(axes_flat[-1])

        # Set x-label only to bottom row plots
        for i in range(len(axes_flat)):
            if i >= len(axes_flat) - 4:
                axes_flat[i].set_xlabel('Time (s)', fontsize=20)
                            # Set custom x-ticks and labels for this subplot
                xtick_positions = np.arange(0, 1.05, 0.2)  # Tick positions every 0.2 seconds
                xtick_labels = [f"{tick:.1f}" for tick in xtick_positions]  # Labels as strings
                ax.set_xticks(xtick_positions)
                ax.set_xticklabels(xtick_labels)
        
        self.figure.tight_layout(h_pad=0.35)
        self.canvas.draw_idle()