from ui_interface import Ui_MainWindow
from PySide6.QtWidgets import QApplication, QMainWindow

import sys
import os

if __name__ == "__main__":
    
    images = os.listdir("images/")
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    
    ui.setupUi(MainWindow, images)
    MainWindow.show()
    
    sys.exit(app.exec())