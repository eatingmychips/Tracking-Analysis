# app/tabs/tab2.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, QHBoxLayout, QLineEdit
from PyQt6.QtCore import QThread, pyqtSignal
from tracking import core_tracking

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

class Project2Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pose_data_list = []
        self.selected_directory = None
        self.init_ui()
        self.worker = None  # Thread to run main loop

    def init_ui(self):
        layout = QVBoxLayout()

        # Combobox for COM port selection
        hbox = QHBoxLayout()