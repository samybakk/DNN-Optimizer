from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QDialog,QTreeView,QFileSystemModel
from PyQt5.QtCore import QDir


class ZipSelector(QDialog) :
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Selector")
        uic.loadUi('Ui/zip_selector.ui', self)

        self.View_Model = self.findChild(QTreeView, 'ModelView')

        self.dirmodel = QFileSystemModel()
        self.dirmodel.setRootPath(QDir.rootPath())
        self.View_Model.setModel(self.dirmodel)
        self.View_Model.setRootIndex(self.dirmodel.index('./Models/Temp/'))
        self.View_Model.header().hideSection(1)
        self.View_Model.header().hideSection(2)
        self.View_Model.header().hideSection(3)
        
        
    def getSelected(self):
        return self.dirmodel.filePath(self.View_Model.currentIndex())
        
        