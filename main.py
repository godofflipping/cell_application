from ui_interface import Ui_MainWindow, QCsMainWindow
from PySide6.QtWidgets import QApplication

import sys


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    MainWindow = QCsMainWindow()
    ui = Ui_MainWindow(MainWindow)
    
    sys.exit(app.exec())