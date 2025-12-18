# app/mainwindow.py
from PyQt6.QtWidgets import QMainWindow, QApplication, QTabWidget, QWidget
import sys

# Import your tab classes
from app.tabs.tab1 import Project1Tab
from app.tabs.tab2 import Project2Tab  
from app.tabs.tab3 import Project3Tab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combined Project GUI")
        self.resize(1080, 720)
        
        # Create the tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Add tabs
        self.tabs.addTab(Project1Tab(self), "Tracking (IR BKPK)")
        self.tabs.addTab(Project3Tab(self), "Tracking (Wireless BKPK)")
        self.tabs.addTab(Project2Tab(self), "Run Analysis")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
