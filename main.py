import sys
from PyQt6.QtWidgets import QApplication
from app.mainwindow import MainWindow
import os 

pylon_x64 = r"C:\Program Files\Basler\pylon\Runtime\x64"
pylon_win32 = r"C:\Program Files\Basler\pylon\Runtime\Win32"

path = os.environ.get("PATH", "")
os.environ["PATH"] = os.pathsep.join([pylon_x64, pylon_win32, path])
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())



