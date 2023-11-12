import os

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect
from PySide6.QtGui import QPixmap, QAction, QImage
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMenu,
    QMenuBar, QSizePolicy, QStatusBar, QVBoxLayout, QWidget, QFileDialog)

from image_viewer import ImageViewer
from watershed import watershedAlgo, dummyAlgo


class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        
        self.cells = dict()
        self.current_cells = []
        self.current_img = ""
        self.algorithm = dummyAlgo
        self.algorithms = dict()
        self.images = []
        self.segment_images = dict()
        self.path_to_dir = ""
        self.is_segmented = False
        self.cell_img_width = 0
        self.cell_img_height = 0
        self.image_width = 1280
        self.image_height = 960
        
        
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
        self.menuHelp.setObjectName(u"menu_algo")
        
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"status_bar")
        
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menubar.addAction(self.menuAlgo.menuAction())
        
        self.load_images = QAction(MainWindow)
        self.load_images.setObjectName(u"load_images")
        self.load_images.triggered.connect(self.getImages)
        self.load_images.setCheckable(False)
        self.menuFile.addAction(self.load_images)
        
        self.segmentation = QAction(MainWindow)
        self.segmentation.setObjectName(u"segmentation")
        self.segmentation.triggered.connect(self.changeMode)
        self.segmentation.setCheckable(False)
        self.menuEdit.addAction(self.segmentation)
        
        self.watershed_option = QAction(MainWindow)
        self.watershed_option.setObjectName(u"watershed")
        self.algorithms[self.watershed_option.objectName()] = watershedAlgo
        self.watershed_option.triggered.connect(self.setWatershedAlgo)
        self.watershed_option.setCheckable(True)
        self.menuAlgo.addAction(self.watershed_option)
        
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
        
        self.img_list = QListWidget(self.centralwidget)
        self.img_list.setObjectName(u"img_list")
        
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.img_list.sizePolicy().hasHeightForWidth())
        
        self.img_list.setSizePolicy(sizePolicy1)
        
        self.verticalLayout.addWidget(self.img_list)

        self.cell_list = QListWidget(self.centralwidget)
        self.cell_list.setObjectName(u"cell_list")
        
        self.img_list.itemClicked.connect(self.getCellList)
        self.img_list.itemClicked.connect(self.changeImage)
        self.img_list.itemClicked.connect(self.getFullInfo)
        
        self.cell_list.itemClicked.connect(self.changeCell)
        self.cell_list.itemClicked.connect(self.getProba)
        self.cell_list.itemClicked.connect(self.getComment)
        
        sizePolicy1.setHeightForWidth(self.cell_list.sizePolicy().hasHeightForWidth())
        
        self.cell_list.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.cell_list)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 3)

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

        self.proba_comm = QLabel(self.centralwidget)
        self.proba_comm.setObjectName(u"proba_comm")
        sizePolicy.setHeightForWidth(self.proba_comm.sizePolicy().hasHeightForWidth())
        self.proba_comm.setSizePolicy(sizePolicy)

        self.verticalLayout_3.addWidget(self.proba_comm)

        self.edit_comm = QLabel(self.centralwidget)
        self.edit_comm.setObjectName(u"edit_comm")
        sizePolicy.setHeightForWidth(self.edit_comm.sizePolicy().hasHeightForWidth())
        self.edit_comm.setSizePolicy(sizePolicy)

        self.verticalLayout_3.addWidget(self.edit_comm)

        self.verticalLayout_3.setStretch(0, 3)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 4)

        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 9)
        self.horizontalLayout.setStretch(2, 4)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        
        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    
    def retranslateUi(self, MainWindow):
        
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Test", None))
        
        self.full_info.setText(QCoreApplication.translate("MainWindow", u"FULL INFO", None))
        self.proba_comm.setText(QCoreApplication.translate("MainWindow", u"PROBABILITY", None))
        self.edit_comm.setText(QCoreApplication.translate("MainWindow", u"EDIT COMMENT", None))
        
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        self.menuAlgo.setTitle(QCoreApplication.translate("MainWindow", u"Algo", None))
        
        self.watershed_option.setText(QCoreApplication.translate("MainWindow", u"watershed", None))
        self.load_images.setText(QCoreApplication.translate("MainWindow", u"Load Images", None))
        self.segmentation.setText(QCoreApplication.translate("MainWindow", u"Segmentation", None))
    
    
    def getImages(self, item):
        if not item:
            self.path_to_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\', QFileDialog.ShowDirsOnly)  + '/'
            self.images = os.listdir(self.path_to_dir)
            self.img_list.clear()
            self.img_list.addItems(self.images)
    
        
    def changeImage(self, item):
        self.current_img = item.text()
        self.is_segmented = False
        self.whole_img.removeBoundaries()
        self.whole_img.setPhoto(QPixmap(self.path_to_dir + item.text()))
        
    
    def cellImageCoords(self, x1, y1, x2, y2):
        
        self.cell_img_width = 200
        self.cell_img_height = 200
        
        x_tl = (x1 + x2) // 2 - 100
        y_tl = (y1 + y2) // 2  - 100
        
        if x_tl <= 0:
            x_tl = 0
        
        if y_tl <= 0:
            y_tl = 0
            
        if y_tl > self.image_width - self.cell_img_width:
            y_tl = self.image_width - self.cell_img_width
            
        if x_tl > self.image_height - self.cell_img_height:
            x_tl = self.image_height - self.cell_img_height
                
        return y_tl, x_tl, self.cell_img_width, self.cell_img_height
    
        
    def changeCell(self, item):
        number = int(''.join(x for x in item.text() if x.isdigit())) - 1
        x1, y1, x2, y2 = self.current_cells[number]
        x, y, width, height = self.cellImageCoords(x1, y1, x2, y2)
        self.cell_img.setPhoto(QPixmap(self.path_to_dir + self.current_img).copy(x, y, width, height))
        self.whole_img.addBoundaries(x, y, width, height)
        
    
    def getCells(self, images):
        for i in range(len(images)):
            image_path = self.path_to_dir + images[i]
            self.cells[images[i]], image = self.algorithm(image_path)
            image = QImage(image, image.shape[1],\
                            image.shape[0], image.shape[1] * 3,QImage.Format_BGR888)
            self.segment_images[images[i]] = QPixmap(image)
        self.cell_list.clear()
        
    
    def getCellList(self, item):
        self.cell_list.clear()
        if item.text() in self.cells.keys():
            self.current_cells = self.cells[item.text()]
            cell_names = ["Cell " + str(i) for i in range(1, len(self.current_cells) + 1)]
            self.cell_list.addItems(cell_names)
               
            
    def changeMode(self, item):
        if not item:
            self.is_segmented = not self.is_segmented
        if self.is_segmented:
            self.whole_img.setPhoto(self.segment_images[self.current_img])
        else:
            self.whole_img.setPhoto(QPixmap(self.path_to_dir + self.current_img))
    
    
    def setWatershedAlgo(self, item):
        if item:
            self.algorithm = self.algorithms['watershed']
            self.getCells(self.images)
       
    
    def getFullInfo(self, item):
        self.full_info.setText(u"You have chosen " + item.text())
        
    
    def getProba(self, item):
        self.proba_comm.setText(u"Probability of " + item.text())
        
        
    def getComment(self, item):
        self.edit_comm.setText(u"Edit comment to " + item.text())
    