from PySide6 import QtCore, QtWidgets

class ListWidget(QtWidgets.QListWidget, QtWidgets.QWidget):
    keyPressed = QtCore.Signal(QtCore.QEvent)
    
    def __init__(self, widget):
        super(ListWidget, self).__init__()
    
    def keyPressEvent(self, event):
        super(ListWidget, self).keyPressEvent(event)
        self.keyPressed.emit(event)