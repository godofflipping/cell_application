import os
import json
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms as tfs
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect, Qt,
     QIODevice, QFile)
from PySide6.QtGui import QPixmap, QAction, QImage
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMenu, QMenuBar,
     QSizePolicy, QStatusBar, QVBoxLayout, QWidget, QFileDialog,
     QPushButton)

from image_viewer import ImageViewer
from list_widget import ListWidget
from watershed import watershedAlgo, dummyAlgo
from main_window import QCsMainWindow
from server_connect import ServerClient


class Ui_MainWindow(QCsMainWindow):
    def __init__ (self, MainWidnow):
        super(Ui_MainWindow, self).__init__()
        MainWidnow.exit_function = self.saveData

        self.startOver()

        self.algorithm = dummyAlgo
        self.algorithms = dict()

        self.server_mode = False
        self.server_client = ServerClient()

        self.first_download = dict()
        self.delete_file_mode = False

        self.setupUi(MainWidnow)
        MainWidnow.show()


    def startOver(self):
        self.cells = dict()
        self.masks = dict()

        self.model = None

        self.current_image_path = ""
        self.current_cell = QPixmap()
        self.current_image = QPixmap()
        self.current_mask = []

        self.images = []
        self.segment_images = dict()

        self.path_to_dir = ""
        self.extentions = ('.jpg', '.jpeg', '.png')
        self.big_extentions = ('.svs')

        self.algo_name = 'dummyAlgo'

        self.is_segmented = False
        self.is_dataset_mode = False
        self.is_algo_chosen = False
        self.is_segment_done = False
        self.is_sorted = False
        self.is_save_mode = False
        self.is_transparent = False
        self.is_model_loaded = False

        self.transparency_const = 0.6

        self.classes = []
        self.path_to_classes = 'data'
        self.path_to_images = 'data'

        self.current_json = ""

        self.cell_img_width = 200
        self.cell_img_height = 200
        self.image_width = 1280
        self.image_height = 960
        self.delta_left_x = 0
        self.delta_right_x = 0
        self.delta_top_y = 0
        self.delta_buttom_y = 0

        self.start_config = {"algorithm": self.algo_name, "cells": {}, "meta_data": {"cells": {}, "coords": [], "masks": []}}
        self.full_data = self.start_config
        self.data_images = dict()

# Создание интерфеса
    def setupUi(self, MainWindow):

        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1080, 720)

        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 957, 22))

        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menu_file")

        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName(u"menu_edit")

        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menu_help")

        self.menuAlgo = QMenu(self.menubar)
        self.menuAlgo.setObjectName(u"menu_algo")

        self.menuClassify = QMenu(self.menubar)
        self.menuClassify.setObjectName(u"menu_classifiy")

        self.menuMode = QMenu(self.menubar)
        self.menuMode.setObjectName(u"menu_mode")

        self.menuServer = QMenu(self.menubar)
        self.menuServer.setObjectName(u"menu_server")

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"status_bar")

        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menubar.addAction(self.menuAlgo.menuAction())
        self.menubar.addAction(self.menuClassify.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())
        self.menubar.addAction(self.menuServer.menuAction())

        self.load_image = QAction(MainWindow)
        self.load_image.setObjectName(u"load_image")
        self.load_image.triggered.connect(self.getImage)
        self.load_image.setCheckable(False)
        self.menuFile.addAction(self.load_image)

        self.load_images = QAction(MainWindow)
        self.load_images.setObjectName(u"load_images")
        self.load_images.triggered.connect(self.getImages)
        self.load_images.setCheckable(False)
        self.menuFile.addAction(self.load_images)

        self.segmentation = QAction(MainWindow)
        self.segmentation.setObjectName(u"segmentation")
        self.segmentation.triggered.connect(self.changeSegmentMode)
        self.segmentation.setCheckable(False)
        self.menuEdit.addAction(self.segmentation)

        self.watershed_option = QAction(MainWindow)
        self.watershed_option.setObjectName(u"watershed")
        self.algorithms[self.watershed_option.objectName()] = watershedAlgo
        self.watershed_option.triggered.connect(self.setWatershedAlgo)
        self.watershed_option.setCheckable(True)
        self.menuAlgo.addAction(self.watershed_option)

        self.effnet_v2 = QAction(MainWindow)
        self.effnet_v2.setObjectName(u"EfficientNetV2")
        self.effnet_v2.triggered.connect(self.predictEffnetV2)
        self.effnet_v2.setCheckable(True)
        self.menuClassify.addAction(self.effnet_v2)

        self.transparency_mode = QAction(MainWindow)
        self.transparency_mode.setObjectName(u"transparency_mode")
        self.transparency_mode.triggered.connect(self.changeTransparencyMode)
        self.transparency_mode.setCheckable(True)
        self.menuMode.addAction(self.transparency_mode)

        self.save_cells_mode = QAction(MainWindow)
        self.save_cells_mode.setObjectName(u"save_cells_mode")
        self.save_cells_mode.triggered.connect(self.changeSaveCellMode)
        self.save_cells_mode.setCheckable(True)
        self.menuMode.addAction(self.save_cells_mode)

        self.server_client_mode = QAction(MainWindow)
        self.server_client_mode.setObjectName(u"dataset_mode")
        self.server_client_mode.triggered.connect(self.changeServerMode)
        self.server_client_mode.setCheckable(True)
        self.menuMode.addAction(self.server_client_mode)

        self.server_upload = QAction(MainWindow)
        self.server_upload.setObjectName(u"dataset_mode")
        self.server_upload.triggered.connect(self.uploadData)
        self.server_upload.setCheckable(False)
        self.menuServer.addAction(self.server_upload)

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())

        self.centralwidget.setSizePolicy(sizePolicy)

        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")

        self.img_list = ListWidget(self.centralwidget)
        self.img_list.setObjectName(u"img_list")

        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.img_list.sizePolicy().hasHeightForWidth())

        self.img_list.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.img_list)

        self.cell_list = ListWidget(self.centralwidget)
        self.cell_list.setObjectName(u"cell_list")

        self.sort_button = QPushButton(self.centralwidget)
        self.sort_button.setObjectName(u"sort_button")
        self.sort_button.clicked.connect(self.sortCellList)
        sizePolicy.setHeightForWidth(self.sort_button.sizePolicy().hasHeightForWidth())
        self.sort_button.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.sort_button)

        self.img_list.itemClicked.connect(self.saveData)
        self.img_list.itemClicked.connect(self.openData)
        self.img_list.itemClicked.connect(self.getCellList)
        self.img_list.itemClicked.connect(self.changeImage)
        self.img_list.itemClicked.connect(self.getFullInfo)

        self.img_list.keyPressed.connect(self.openData)
        self.img_list.keyPressed.connect(self.getCellListFromKey)
        self.img_list.keyPressed.connect(self.changeImageFromKey)
        self.img_list.keyPressed.connect(self.getFullInfoFromKey)

        self.cell_list.itemClicked.connect(self.changeCell)
        self.cell_list.itemClicked.connect(self.getProba)
        self.cell_list.itemClicked.connect(self.changeModeCellPerm)
        #self.cell_list.itemClicked.connect(self.changeModePerm)
        self.cell_list.itemClicked.connect(self.getFullInfo)

        self.cell_list.keyPressed.connect(self.changeCellFromKey)
        self.cell_list.keyPressed.connect(self.getProbaFromKey)
        self.cell_list.keyPressed.connect(self.changeModePerm)
        self.cell_list.keyPressed.connect(self.getFullInfoFromKey)

        sizePolicy1.setHeightForWidth(self.cell_list.sizePolicy().hasHeightForWidth())

        self.cell_list.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.cell_list)

        self.verticalLayout.setStretch(0, 8)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 24)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")

        self.whole_img = ImageViewer(self.centralwidget)
        self.whole_img.setObjectName(u"whole_img")

        self.verticalLayout_2.addWidget(self.whole_img)

        self.full_info = QLabel(self.centralwidget)
        self.full_info.setObjectName(u"full_info")
        sizePolicy.setHeightForWidth(self.full_info.sizePolicy().hasHeightForWidth())
        self.full_info.setSizePolicy(sizePolicy)

        self.verticalLayout_2.addWidget(self.full_info)

        self.verticalLayout_2.setStretch(0, 4)
        self.verticalLayout_2.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")

        self.cell_img = ImageViewer(self.centralwidget)
        self.cell_img.setObjectName(u"cell_img")

        self.verticalLayout_3.addWidget(self.cell_img)

        self.edit_button = QPushButton(self.centralwidget)
        self.edit_button.setObjectName(u"edit_button")
        self.edit_button.clicked.connect(self.changeMode)
        sizePolicy.setHeightForWidth(self.edit_button.sizePolicy().hasHeightForWidth())
        self.edit_button.setSizePolicy(sizePolicy)

        self.verticalLayout_3.addWidget(self.edit_button)

        self.proba_comm = QLabel(self.centralwidget)
        self.proba_comm.setObjectName(u"proba_comm")
        sizePolicy.setHeightForWidth(self.proba_comm.sizePolicy().hasHeightForWidth())
        self.proba_comm.setSizePolicy(sizePolicy)

        self.verticalLayout_3.addWidget(self.proba_comm)

        self.verticalLayout_3.setStretch(0, 3)
        self.verticalLayout_3.setStretch(1, 0.5)
        self.verticalLayout_3.setStretch(2, 5)

        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 9)
        self.horizontalLayout.setStretch(2, 4)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Application", None))

        self.full_info.setText(QCoreApplication.translate("MainWindow", u"FULL INFO", None))
        self.proba_comm.setText(QCoreApplication.translate("MainWindow", u"CLASS (PROBABILITY)", None))

        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        self.menuAlgo.setTitle(QCoreApplication.translate("MainWindow", u"Algorithms", None))
        self.menuClassify.setTitle(QCoreApplication.translate("MainWindow", u"Classify", None))
        self.menuMode.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.menuServer.setTitle(QCoreApplication.translate("MainWindow", u"Server", None))

        self.watershed_option.setText(QCoreApplication.translate("MainWindow", u"watershed", None))
        self.load_image.setText(QCoreApplication.translate("MainWindow", u"Load Image", None))
        self.load_images.setText(QCoreApplication.translate("MainWindow", u"Load Images", None))
        self.segmentation.setText(QCoreApplication.translate("MainWindow", u"Segmentation", None))
        self.effnet_v2.setText(QCoreApplication.translate("MainWindow", u"EfficientNetV2", None))
        self.transparency_mode.setText(QCoreApplication.translate("MainWindow", u"Transparency Mode", None))
        self.save_cells_mode.setText(QCoreApplication.translate("MainWindow", u"Save Cells Mode", None))
        self.server_client_mode.setText(QCoreApplication.translate("MainWindow", u"Server Mode", None))
        self.server_upload.setText(QCoreApplication.translate("MainWindow", u"Server Upload", None))
        self.edit_button.setText(QCoreApplication.translate("MainWindow", u"Change Class", None))
        self.sort_button.setText(QCoreApplication.translate("MainWindow", u"Sort Cells", None))

    def changeServerMode(self, item):
        if item:
            self.server_mode = True
        else:
            self.server_mode = False

    def changeSaveCellMode(self, item):
        if item:
            self.is_save_mode = True
        else:
            self.is_save_mode = False

    def changeTransparencyMode(self, item):
        if item:
            self.is_transparent = True
        else:
            self.is_transparent = False
        try:
            self.changeCell(self.cell_list.currentItem().text())
        except:
            pass

    def getImage(self, item):
        if not item:
            self.images.clear()
            self.cells.clear()

            path, _ = QFileDialog.getOpenFileName(None, 'Open File', './', "Image (*.png *.jpg *.jpeg)")
            img = path.split('/')[-1]
            self.path_to_dir = path.replace(img, '')

            if img.endswith(self.extentions) or img.endswith(self.big_extentions):
                    self.images.append(img)

            self.img_list.clear()
            self.img_list.addItems(self.images)

    # Получение папки с изображениями (File -> Load Images)
    def getImages(self, item):
        if not item:
            self.images.clear()
            self.cells.clear()
            self.path_to_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\', QFileDialog.ShowDirsOnly)  + '/'
            files = os.listdir(self.path_to_dir)

            for img in files:
                if img.endswith(self.extentions) or img.endswith(self.big_extentions):
                    self.images.append(img)

            self.img_list.clear()
            self.img_list.addItems(self.images)


    # Проверка нажатия определённой кнопки на клавиатуре
    def keyCheck(self, event):
        return event.key() == Qt.Key_Return


###############################################################################

    # Смена основного изображения
    def changeImage(self, item):
        self.current_image_path = item.text()
        self.is_segmented = False
        self.whole_img.removeBoundaries()

        self.changeModePerm(item=None)
        self.current_image = QPixmap(self.path_to_dir + item.text())

        if self.current_image_path.endswith(self.extentions):
            self.whole_img.setPhoto(self.current_image)

        if item.text() in self.segment_images:
            image_size = self.segment_images[item.text()][1]
            self.image_height = image_size[0]
            self.image_width = image_size[1]


    # Смена основного изображения через клавиатуру
    def changeImageFromKey(self, event):
        if self.keyCheck(event):
            self.changeImage(self.img_list.currentItem())


###############################################################################


    # Нахождение центра клетки (после получения bounding box)
    # и левых верхних координат изображения клетки
    def cellImageCoords(self, x1, y1, x2, y2):
        x_tl = (x1 + x2 - self.cell_img_width) // 2
        y_tl = (y1 + y2  - self.cell_img_height) // 2

        if x_tl <= 0:
            x_tl = 0

        if y_tl <= 0:
            y_tl = 0

        if y_tl > self.image_width - self.cell_img_width:
            y_tl = self.image_width - self.cell_img_width

        if x_tl > self.image_height - self.cell_img_height:
            x_tl = self.image_height - self.cell_img_height

        self.delta_top_y = x1 - x_tl
        self.delta_buttom_y = self.cell_img_width + x_tl - x2
        self.delta_left_x = y1 - y_tl
        self.delta_right_x = self.cell_img_height + y_tl - y2

        return y_tl, x_tl, self.cell_img_width, self.cell_img_height


    def getCellMask(self, number):
        mask = []
        false_full_line = ""
        false_left_line = ""
        false_right_line = ""

        for i in range(self.cell_img_height):
            false_full_line += "0"
        for i in range(self.delta_left_x):
            false_left_line += "0"
        for i in range(self.delta_right_x):
            false_right_line += "0"

        for i in range(self.delta_top_y):
            mask.append(false_full_line)
        for i in range(self.cell_img_height - self.delta_buttom_y - self.delta_top_y):
            mask.append(false_left_line + self.full_data['meta_data']['masks'][number][i] + false_right_line)
        for i in range(self.delta_buttom_y):
            mask.append(false_full_line)

        result = []
        for current_line in mask:
            result.append([int(i) for i in current_line])

        return np.array(result)


###############################################################################


    # Смена изображения клетки
    def changeCell(self, item):
        number = int(self.cell_list.currentItem().text().split(' ')[1]) - 1
        x1, y1, x2, y2 = self.full_data['meta_data']['coords'][number]
        x, y, width, height = self.cellImageCoords(x1, y1, x2, y2)

        self.current_cell = self.current_image.copy(x, y, width, height)

        if self.is_transparent:
            self.current_mask = self.getCellMask(number)
            qimage = self.current_cell.toImage().convertToFormat(QImage.Format_RGB888)
            cell_array = np.ndarray((qimage.height(), qimage.width(), 3), buffer=qimage.constBits(), strides=[qimage.bytesPerLine(), 3, 1], dtype=np.uint8)
            array = cell_array.copy()
            array[self.current_mask == 0] = cell_array[self.current_mask == 0] * self.transparency_const
            self.current_cell = QPixmap()
            self.current_cell.convertFromImage(QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format_RGB888))

        self.cell_img.setPhoto(self.current_cell)
        self.whole_img.addBoundaries(x, y, width, height)


    # Смена изображения клетки через клавиатуру
    def changeCellFromKey(self, event):
        if self.keyCheck(event):
            self.changeCell(self.cell_list.currentItem())


###############################################################################


    # Получение отсегментированного изображения и bounding box клеток
    def getCells(self, images):
        try:
            if self.algo_name != 'dummyAlgo':
                for i in range(len(images)):
                    image_path = self.path_to_dir + images[i]
                    self.cells[images[i]], image, self.masks[images[i]] = self.algorithm(image_path)

                    self.segment_images[images[i]] = image, image.shape

            else:
                self.cells[self.img_list.currentItem().text()] = self.full_data['meta_data']['coords']
                self.masks[self.img_list.currentItem().text()] = self.full_data['meta_data']['masks']

            self.cell_list.clear()
        except:
            pass


###############################################################################


    # Получение списка всех клеток
    def getCellList(self, item):
        if self.is_algo_chosen and not self.is_segment_done:
            self.getCells(self.images)
            self.is_segment_done = True

        if self.full_data['algorithm'] != 'dummyAlgo':
            self.getCells(self.images)

        self.cell_list.clear()

        if item.text() in self.cells:

            self.full_data['meta_data']['coords'] = self.cells[item.text()]
            self.full_data['meta_data']['masks'] = self.masks[item.text()]

            self.full_data['meta_data']['coords'] = self.cells[item.text()]
            self.full_data['meta_data']['masks'] = self.masks[item.text()]

            cell_names = ["Cell " + str(i) for i in range(1, len(self.full_data['meta_data']['coords']) + 1)]

            for key in self.full_data['meta_data']['cells']:
                cell_index = int(key.split('_')[1]) - 1
                cell_class = max(self.full_data['meta_data']['cells'][key]["probability"], key = self.full_data['meta_data']['cells'][key]["probability"].get)
                cell_proba = self.full_data['meta_data']['cells'][key]["probability"][cell_class]
                cell_names[cell_index] += (' ' + cell_class + ' ' + str(round(cell_proba, 3)))

            self.cell_list.addItems(cell_names)


    # Получение списка всех клеток через клавиатуру
    def getCellListFromKey(self, event):
        if self.keyCheck(event):
            self.getCellList(self.img_list.currentItem())


    # Сортировка списка клеток
    def sortCellList(self):
        self.is_sorted = not self.is_sorted
        items = [self.cell_list.item(item).text() for item in range(self.cell_list.count())]
        self.cell_list.clear()

        if self.is_sorted:
            none_class = [item for item in items if len(item.split(' ')) == 2]
            with_class = [item for item in items if len(item.split(' ')) > 2]

            with_class.sort(key = lambda x: (
                x.split(' ')[2],
                x.split(' ')[3],
                int(x.split(' ')[1])
            ))
            none_class.sort(key = lambda x: int(x.split(' ')[1]))

            self.cell_list.addItems(with_class)
            self.cell_list.addItems(none_class)
        else:
            items.sort(key = lambda x: int(x.split(' ')[1]))
            self.cell_list.addItems(items)

###############################################################################


    # Отображение сегментированного изображения
    def changeSegmentMode(self, item):
        try:
            if not item:
                self.is_segmented = not self.is_segmented

            if self.is_segmented and self.is_algo_chosen:
                image = self.segment_images[self.current_image_path][0]
                image = QImage(image, image.shape[1],\
                                    image.shape[0], image.shape[1] * 3, QImage.Format_BGR888)
                self.whole_img.setPhoto(QPixmap(image))

            else:
                self.whole_img.setPhoto(self.current_image)
        except:
            pass


    # Выбор watershed алгоритма сегментации
    def setWatershedAlgo(self, item):
        if item:
            self.algorithm = self.algorithms['watershed']
            self.is_algo_chosen = True
            self.full_data['algorithm'] = self.algo_name = 'watershed'
        else:
            self.is_algo_chosen = False


    # Добавление названий классов для датасета при их отсутствия в папке
    def getCellClasses(self):
        if len(self.classes) == 0:
            path = "classes.txt"
            with open(path) as file:
                self.classes = [line.strip() for line in file]


    def openJson(self):
        try:
            with open(self.current_json, 'r+') as file:
                self.full_data = json.load(file)
                self.data_images.clear()
        except:
            pass


    # Получение всех элементов датасета
    def openData(self):
        file_name = self.path_to_images + '/' + self.img_list.currentItem().text().split('.')[0]

        self.getCellClasses()

        if not os.path.exists(self.path_to_images):
            os.mkdir(self.path_to_images)

        if not os.path.exists(file_name):
            os.mkdir(file_name)

        json_name = file_name + '/' + self.img_list.currentItem().text().split('.')[0] + ".json"
        self.current_json = json_name

        if self.server_mode and file_name not in self.first_download:
                self.first_download[file_name] = True
                self.downloadJson(json_name)

        if not os.path.exists(json_name):
            file = open(json_name, 'w')
            file.write('{"algorithm": self.algo_name, "cells": {}, "meta_data": {"cells": {}, "coords": [], "masks": []}}')
        else:
            self.openJson()

        self.path_to_classes = file_name

        if self.is_save_mode:
            for cell_class in self.classes:
                path = self.path_to_classes + '/' + cell_class
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)


    # Сохранение всех элементов датасета в json и в папки классов
    def saveData(self):
        self.is_dataset_mode = False

        if len(self.current_json) != 0:
            try:
                with open(self.current_json, 'w+') as file:
                    json.dump(self.full_data, file)

                if self.is_save_mode:
                    for key in self.data_images.keys():
                        key_name = key[0].split('-')[0]
                        file = QFile(key[1] + '/' + key_name + ".png")
                        file.open(QIODevice.WriteOnly)
                        self.data_images[key].save(file, "PNG")

                if self.server_mode and self.path_to_classes in self.first_download:
                    self.server_client.upload(
                        source_file = self.current_json,
                        destination_file = self.current_json
                    )

                    if self.path_to_classes in self.first_download:
                        if self.first_download[self.path_to_classes]:
                            self.first_download[self.path_to_classes] = False

                            self.server_client.upload(
                                source_file = self.path_to_dir + self.current_image_path,
                                destination_file = self.path_to_classes + '/' + self.current_image_path
                            )

            except:
                pass
        self.full_data['meta_data']['cells'] = self.start_config['cells']
        self.full_data['meta_data'] = self.start_config['meta_data']

###############################################################################


    # Добавление элемента в датасет
    def addCellItem(self, item):
        item_class = item.text().split(' ')[0]

        key = self.img_list.currentItem().text().split('.')[0] + '_' + \
              self.cell_list.currentItem().text().split(' ')[1]

        if self.is_save_mode and key in self.full_data['meta_data']['cells'].keys():
            prev_value = self.full_data['meta_data']['cells'][key]["path"]
            filename = prev_value + '/' + self.img_list.currentItem().text().split('.')[0] + '_' + \
                       self.cell_list.currentItem().text().split(' ')[1] + '.png'
            if os.path.exists(filename):
                os.remove(filename)

        value = self.path_to_classes + '/' + item_class

        probability = dict()
        for cell_class in self.classes:
            if cell_class != item_class:
                probability[cell_class] = 0
            else:
                probability[cell_class] = 1

        meta_data = {
            "path": value,
            "probability": probability,
        }

        self.full_data['meta_data']['cells'][key] = meta_data

        self.data_images[(key, self.full_data['meta_data']['cells'][key]["path"])] = self.current_cell
        self.getProba(self.cell_list.currentItem())


    # Добавление элемента в датасет через клавиатуру
    def addCellItemFromKey(self, event):
        if self.keyCheck(event):
            self.addCellItem(self.proba_write.currentItem())

    def loadEffnetV2(self):
        self.is_model_loaded = True
        model_path = 'model/EfficientNetV2/TransferLearning/2layer1linear/weights.pth'
        self.model = torchvision.models.efficientnet_v2_s(pretrained=True)

        for param in self.model.features[:-2].parameters():
            param.require_grad = False

        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(num_features, len(self.classes))

        self.model.load_state_dict(torch.load(model_path))
        self.transform = tfs.Compose([
            tfs.Resize(224),
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def predictEffnetV2(self, item):
        if item:
            if not self.is_model_loaded:
                self.loadEffnetV2()

            image = Image.open(self.path_to_dir + self.current_image_path)

            for i in range(len(self.full_data['meta_data']['coords'])):
                probability = dict()
                x1, y1, x2, y2 = self.full_data['meta_data']['coords'][i]
                x, y, width, height = self.cellImageCoords(x1, y1, x2, y2)

                cell_image = image.crop((x, y, x + width, y + height))
                cell_image = self.transform(cell_image)
                cell_image = cell_image.view(1, 3, 224, 224)

                output = self.model(cell_image)
                prediction = torch.softmax(output, dim=1).detach().numpy()

                for j, cell_class in enumerate(self.classes):
                    probability[cell_class] = prediction[0][j].astype('float64')

                #######################
                meta_data = {
                    "path": i,
                    "probability": probability,
                }

                key = self.img_list.currentItem().text().split('.')[0] + '_' + str(i + 1)
                self.full_data['meta_data']['cells'][key] = meta_data
            print("DONE")

###############################################################################
    # Возможность формировать датасет
    def changeMode(self, item):
        try:
            if not item:
                self.is_dataset_mode = not self.is_dataset_mode

            if self.is_dataset_mode:

                self.edit_button.setText(u'Confirm')
                self.proba_write = ListWidget(self.centralwidget)
                self.proba_write.setObjectName(u"class_list")

                key = self.img_list.currentItem().text().split('.')[0] + '_' + \
                    self.cell_list.currentItem().text().split(' ')[1]

                if key in self.full_data['meta_data']['cells'].keys():
                    for cell_class in self.classes:
                        self.proba_write.addItem(cell_class + " (" + \
                        str(self.full_data['meta_data']['cells'][key]["probability"][cell_class]) + ")")
                else:
                    self.proba_write.addItems(self.classes)

                self.proba_write.itemClicked.connect(self.addCellItem)
                self.proba_write.keyPressed.connect(self.addCellItemFromKey)
                self.verticalLayout_3.replaceWidget(self.proba_comm, self.proba_write)

            else:
                self.edit_button.setText(u'Change Class')
                self.verticalLayout_3.replaceWidget(self.proba_write, self.proba_comm)
                self.proba_write.deleteLater()

        except:
            pass


    def changeModePerm(self, item):
        self.is_dataset_mode = False
        try:
            self.saveData()
            self.openData()
            self.edit_button.setText(u'Change Class')
            self.verticalLayout_3.replaceWidget(self.proba_write, self.proba_comm)
            self.proba_write.deleteLater()
        except:
            pass


    def changeModeCellPerm(self, item):
        self.is_dataset_mode = False
        try:
            self.edit_button.setText(u'Change Class')
            self.verticalLayout_3.replaceWidget(self.proba_write, self.proba_comm)
            self.proba_write.deleteLater()
        except:
            pass


    def changeModeCellFromKey(self, event):
        if self.keyCheck(event):
            self.changeModeCellPerm(self.cell_list.currentItem())


    def changeModePermFromKey(self, event):
        if self.keyCheck(event):
            self.changeModePerm(self.cell_list.currentItem())


###############################################################################


    # Отображение полной информации об основном изображении
    def getFullInfo(self, item):
        full_text = "Total amount: "

        if len(self.full_data['meta_data']['coords']) != 0:
            full_text += str(len(self.full_data['meta_data']['coords'])) + '\n'
        else:
            full_text += "0" + '\n'

        count_class = dict()

        for cell_class in self.classes:
            count_class[cell_class] = 0
            for key in self.full_data['meta_data']['cells'].keys():
                if cell_class == max(self.full_data['meta_data']['cells'][key]["probability"], key = self.full_data['meta_data']['cells'][key]["probability"].get):
                    count_class[cell_class] += 1

        counter = 1
        for cell_class in self.classes:

            if len(self.full_data['meta_data']['cells']) != 0:
                full_text += cell_class + "\t" + str(count_class[cell_class])
            else:
                full_text += cell_class + "\t0"

            if counter % 4 == 0:
                full_text += "\n"
            else:
                full_text += "\t"

            counter += 1

        self.full_info.setText(full_text)


    # Отображение полной информации об основном изображении через клавиатуру
    def getFullInfoFromKey(self, event):
        if self.keyCheck(event):
            self.getFullInfo(self.img_list.currentItem())


###############################################################################


    # Оторажение верояности или класса (при формировании датасета)
    def getProba(self, item):
        key = self.img_list.currentItem().text().split('.')[0] + '_' + \
              item.text().split(' ')[1]
        if key not in self.full_data['meta_data']['cells'].keys():
            self.proba_comm.setText(u"CLASS (PROBABILITY)")
        else:
            cell_class = max(self.full_data['meta_data']['cells'][key]["probability"], key = self.full_data['meta_data']['cells'][key]["probability"].get)
            cell_proba = round(self.full_data['meta_data']['cells'][key]["probability"][cell_class], 3)
            self.proba_comm.setText(cell_class + " (" + str(cell_proba) + ")")


    # Оторажение верояности или класса (при формировании датасета) через клавиатуру
    def getProbaFromKey(self, event):
        if self.keyCheck(event):
            self.getProba(self.cell_list.currentItem())


###############################################################################


    def uploadData(self, item):
        for path, subdirs, files in os.walk(self.path_to_images):
            for name in files:
                file_path = os.path.join(path, name).replace("\\", "/")

                if file_path.endswith('.json') or self.is_save_mode:
                    self.server_client.upload(
                        source_file = file_path,
                        destination_file = file_path
                    )


    def downloadJson(self, file_name):
        self.server_client.download(
            source_file = file_name,
            download_file = file_name
        )
