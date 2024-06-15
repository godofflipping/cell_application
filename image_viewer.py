from PySide6.QtCore import QPointF, Signal, Qt, QRectF, QLineF
from PySide6.QtGui import QPixmap, QBrush, QColor, QPen
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene,
                               QGraphicsPixmapItem,
                               QFrame, QGraphicsRectItem)


# Отображение изображения с возможностью масштабирования
class ImageViewer(QGraphicsView):
    photoClicked = Signal(QPointF)

    def __init__(self, parent):
        super(ImageViewer, self).__init__(parent)

        self.zoom = 0
        self.empty = True

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pen = QPen(Qt.green)

        self.rect = QGraphicsRectItem()
        self.addRectToScene(0, 0, 0, 0)

        self.photo = QGraphicsPixmapItem()
        self.scene.addItem(self.photo)

        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.Shape.NoFrame)

    def hasPhoto(self):
        return not self.empty

    def fitInView(self, scale=True):
        rect = QRectF(self.photo.pixmap().rect())

        if not rect.isNull():

            self.setSceneRect(rect)

            if self.hasPhoto():

                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)

                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())

                self.scale(factor, factor)

            self.zoom = 0

    def setPhoto(self, pixmap=None, is_big_image=False):
        self.zoom = 0

        if pixmap and not pixmap.isNull():
            self.empty = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.photo.setPixmap(pixmap)

        else:
            self.empty = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.photo.setPixmap(QPixmap())

        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():

            if event.angleDelta().y() > 0:
                factor = 1.25
                self.zoom += 1

            else:
                factor = 0.8
                self.zoom -= 1

            if self.zoom > 0:
                self.scale(factor, factor)

            elif self.zoom == 0:
                self.fitInView()

            else:
                self.zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

        elif not self.photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self.photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.position().toPoint()))

        super(ImageViewer, self).mousePressEvent(event)

    def addRectToScene(self, x1, y1, x2, y2):
        self.rect.setRect(x1, y1, x2-x1, y2-y1)
        self.scene.addItem(self.rect)

    def addBoundaries(self, x, y, width, height):
        self.scene.removeItem(self.rect)
        self.rect.setPen(self.pen)
        self.addRectToScene(x, y, x + width, y + height)

    def removeBoundaries(self):
        self.scene.removeItem(self.rect)
        self.addRectToScene(0, 0, 0, 0)
