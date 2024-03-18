from ui_interface import Ui_MainWindow, QCsMainWindow
from PySide6.QtWidgets import QApplication

import sys, ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
    
def run():
    app = QApplication(sys.argv)
    MainWindow = QCsMainWindow()
    ui = Ui_MainWindow(MainWindow) 
    sys.exit(app.exec())
    
def run_as_admin(pyinstaller):
    fourth = " ".join(sys.argv)
    if pyinstaller:
        fourth = " ".join(sys.argv[1:])
    
    if is_admin():    
        run()
    
    else:
        ctypes.windll.shell32.ShellExecuteW(None, 
            "runas", sys.executable, fourth, None, 1
        )

if __name__ == "__main__":
    exe_file = False
    
    if exe_file:
        run_as_admin(pyinstaller=exe_file)
    else:
        run()