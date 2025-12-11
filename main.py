import sys
from PyQt6.QtWidgets import QApplication
from app.mainwindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


#TODO: Add Comments in to file saving 
#TODO: Add better code commenting 
#TODO: Fix Analysis on NOTHING 


