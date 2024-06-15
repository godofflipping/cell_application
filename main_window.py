from PySide6.QtWidgets import QMainWindow, QWidget, QMessageBox


# QMainWindow с всплывающим окном для закрытия
class QCsMainWindow(QMainWindow):
    def __init__(self):
        super(QCsMainWindow, self).__init__()
        self.isDirectlyClose = False
        self.exit_function = None

    def close(self):
        for childQWidget in self.findChildren(QWidget):
            childQWidget.close()
        self.isDirectlyClose = True
        return QMainWindow.close(self)

    def closeEvent(self, eventQCloseEvent):
        if self.isDirectlyClose:
            eventQCloseEvent.accept()
            self.exit_function()
        else:
            answer = QMessageBox.question(
                self,
                'Save Window',
                'Are you want to exit the application and save your current progress?',
                QMessageBox.Ok,
                QMessageBox.Cancel)
            if (answer == QMessageBox.Ok) or (self.isDirectlyClose == True):
                eventQCloseEvent.accept()
                self.exit_function()
            else:
                eventQCloseEvent.ignore()
