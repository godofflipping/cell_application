import os
import json

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, Qt
from PySide6.QtGui import QPixmap, QAction, QImage
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMenu, QMenuBar, 
    QSizePolicy, QStatusBar, QVBoxLayout, QWidget, QFileDialog,
    QTextEdit)

from image_viewer import ImageViewer
from list_widget import ListWidget
from watershed import watershedAlgo, dummyAlgo
from dataset_creator import DatasetCreator
from main_window import QCsMainWindow
from hash import pHash


class Ui_MainWindow(QCsMainWindow):
    
    def __init__ (self, MainWidnow):
        
        super(Ui_MainWindow, self).__init__()
        MainWidnow.exit_function = self.DSC_saveDataset
        
        self.cells = dict()
        self.current_cells = []
        
        self.current_img = ""
        
        self.algorithm = dummyAlgo
        self.algorithms = dict()
        
        self.images = []
        self.segment_images = dict()
        
        self.path_to_dir = ""
        self.extentions = ('.jpg', '.jpeg', '.png', '.bmp', '.eps')
        
        self.is_segmented = False
        
        self.classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 
                        'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
        self.path_to_classes = 'data/'
        
        self.cell_img_width = 200
        self.cell_img_height = 200
        self.image_width = 1280
        self.image_height = 960
        
        self.data = dict()
        
        self.setupUi(MainWidnow)
        MainWidnow.show()
    
    
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
        
        self.menuMode = QMenu(self.menubar)
        self.menuMode.setObjectName(u"menu_mode")
        
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"status_bar")
        
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menubar.addAction(self.menuAlgo.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())
        
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
        
        self.dataset_mode = QAction(MainWindow)
        self.dataset_mode.setObjectName(u"dataset_mode")
        self.dataset_mode.triggered.connect(self.changeMode)
        self.dataset_mode.setCheckable(True)
        self.menuMode.addAction(self.dataset_mode)
        
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
        
        self.img_list.itemClicked.connect(self.getCellList)
        self.img_list.itemClicked.connect(self.changeImage)
        self.img_list.itemClicked.connect(self.getFullInfo)
        
        self.img_list.keyPressed.connect(self.getCellListFromKey)
        self.img_list.keyPressed.connect(self.changeImageFromKey)
        self.img_list.keyPressed.connect(self.getFullInfoFromKey)
        
        self.cell_list.itemClicked.connect(self.changeCell)
        self.cell_list.itemClicked.connect(self.getProba)
        self.cell_list.itemClicked.connect(self.getComment)
        
        self.cell_list.keyPressed.connect(self.changeCellFromKey)
        self.cell_list.keyPressed.connect(self.getProbaFromKey)
        self.cell_list.keyPressed.connect(self.getCommentFromKey)
        
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
        
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Application", None))
        
        self.full_info.setText(QCoreApplication.translate("MainWindow", u"FULL INFO", None))
        self.proba_comm.setText(QCoreApplication.translate("MainWindow", u"PROBABILITY", None))
        self.edit_comm.setText(QCoreApplication.translate("MainWindow", u"COMMENT FOR DATASET MODE", None))
        
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        self.menuAlgo.setTitle(QCoreApplication.translate("MainWindow", u"Algorithms", None))
        self.menuMode.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
        
        self.watershed_option.setText(QCoreApplication.translate("MainWindow", u"watershed", None))
        self.load_images.setText(QCoreApplication.translate("MainWindow", u"Load Images", None))
        self.segmentation.setText(QCoreApplication.translate("MainWindow", u"Segmentation", None))
        self.dataset_mode.setText(QCoreApplication.translate("MainWindow", u"Dataset Mode", None))
    
    
    def getImages(self, item):
        
        if not item:
            self.path_to_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\', QFileDialog.ShowDirsOnly)  + '/'
            files = os.listdir(self.path_to_dir)
        
            for img in files:
                if img.endswith(self.extentions):
                    self.images.append(img)
                    
            self.img_list.clear()
            self.img_list.addItems(self.images)
            
            
    def keyCheck(self, event):
        return event.key() == Qt.Key_Return
    
    
###############################################################################
    
        
    def changeImage(self, item):
        
        self.current_img = item.text()
        self.is_segmented = False
        self.whole_img.removeBoundaries()
        self.whole_img.setPhoto(QPixmap(self.path_to_dir + item.text()))
        
        if item.text() in self.segment_images:
            _, image_size, _ = self.segment_images[item.text()]
            self.image_height = image_size.height()
            self.image_width = image_size.width()
            
    
    def changeImageFromKey(self, event):
        if self.keyCheck(event):
            self.changeImage(self.img_list.currentItem())
            
            
###############################################################################
        
    
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
                
        return y_tl, x_tl, self.cell_img_width, self.cell_img_height
    
    
###############################################################################
    
        
    def changeCell(self, item):
        number = int(''.join(x for x in item.text() if x.isdigit())) - 1
        x1, y1, x2, y2 = self.current_cells[number]
        x, y, width, height = self.cellImageCoords(x1, y1, x2, y2)
        self.cell_img.setPhoto(QPixmap(self.path_to_dir + self.current_img).copy(x, y, width, height))
        self.whole_img.addBoundaries(x, y, width, height)
        
        
    def changeCellFromKey(self, event):
        if self.keyCheck(event):
            self.changeCell(self.cell_list.currentItem())
            
            
###############################################################################

    
    def getCells(self, images):
        for i in range(len(images)):
            image_path = self.path_to_dir + images[i]
            self.cells[images[i]], image = self.algorithm(image_path)
            image_hash = pHash(image)
            
            image = QImage(image, image.shape[1],\
                            image.shape[0], image.shape[1] * 3, QImage.Format_BGR888)
            self.segment_images[images[i]] = QPixmap(image), image.size(), image_hash
        self.cell_list.clear()
     
     
###############################################################################
        
    
    def getCellList(self, item):
        self.cell_list.clear()
        
        if item.text() in self.cells.keys():
            self.current_cells = self.cells[item.text()]
            cell_names = ["Cell " + str(i) for i in range(1, len(self.current_cells) + 1)]
            self.cell_list.addItems(cell_names)
    
            
    def getCellListFromKey(self, event):
        if self.keyCheck(event):
            self.getCellList(self.img_list.currentItem())
            

###############################################################################

            
    def changeSegmentMode(self, item):
        if not item:
            self.is_segmented = not self.is_segmented
        
        if self.is_segmented:
            self.whole_img.setPhoto(self.segment_images[self.current_img][0])
            
        else:
            self.whole_img.setPhoto(QPixmap(self.path_to_dir + self.current_img))
    
    
    def setWatershedAlgo(self, item):
        if item:
            self.algorithm = self.algorithms['watershed']
            self.getCells(self.images)
            
            
    def DSC_openDataset(self):
        with open('dataset.json', 'r+') as file:
            self.data = json.load(file)
        for cell_class in self.classes:
            path = self.path_to_classes + cell_class
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                
            
    def DSC_saveDataset(self):
        with open('dataset.json', 'w+') as file:
            json.dump(self.data, file)
    

###############################################################################
    
    
    def DSC_addItem(self, item):
        key = self.img_list.currentItem().text() + '/' + \
              self.cell_list.currentItem().text()
        value = self.path_to_classes + item.text()
        self.data.update({key: value})
        self.proba_comm.setText(self.data[key])
        self.edit_comm.setText()
        
        
    
    def DSC_addItemFromKey(self, event):
        if self.keyCheck(event):
            self.DSC_addItem(self.proba_write.currentItem())
            
            
###############################################################################


    def DSC_removeItem(self, index):
        pass
        
        
    def changeMode(self, item):
        if item:
            self.DSC_openDataset()
            self.proba_write = ListWidget(self.centralwidget)
            self.proba_write.setObjectName(u"class_list")
            self.proba_write.addItems(self.classes)
            self.proba_write.itemClicked.connect(self.DSC_addItem)
            self.proba_write.itemClicked.connect(self.getComment)
            self.proba_write.keyPressed.connect(self.DSC_addItemFromKey)
            self.proba_write.keyPressed.connect(self.getCommentFromKey)
            self.verticalLayout_3.replaceWidget(self.proba_comm, self.proba_write)
        
        else:
            if len(self.data) != 0:
                self.DSC_saveDataset()
            self.verticalLayout_3.replaceWidget(self.proba_write, self.proba_comm)
            self.proba_write.deleteLater()
            
    
###############################################################################


    def getFullInfo(self, item):
        self.full_info.setText(u"You have chosen " + item.text())
        
        
    def getFullInfoFromKey(self, event):
        if self.keyCheck(event):
            self.getFullInfo(self.img_list.currentItem())
            
            
###############################################################################        
        

    def getProba(self, item):
        key = self.img_list.currentItem().text() + '/' + item.text()
        if key not in self.data:
            self.proba_comm.setText(u"Probability of " + item.text())
        else:
            self.proba_comm.setText(self.data[key])
        
        
    def getProbaFromKey(self, event):
        if self.keyCheck(event):
            self.getProba(self.cell_list.currentItem())
        
        
###############################################################################
        
        
    def getComment(self, item):
        self.edit_comm.setText(u"Edit comment to " + item.text())
        
        
    def getCommentFromKey(self, event):
        if self.keyCheck(event):
            self.getComment(self.cell_list.currentItem())
            
            
###############################################################################