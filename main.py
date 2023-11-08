from ui_interface import Ui_MainWindow
from PySide6.QtWidgets import QApplication, QMainWindow

import sys

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    
    ui.setupUi(MainWindow)
    MainWindow.show()
    
    sys.exit(app.exec())