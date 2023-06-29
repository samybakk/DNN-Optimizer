
import os
import sys,logging

sys.path.insert(0, './yolov5')
sys.path.insert(0,'./yolov5-knowledge-distillation')
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog,QTextEdit,QPlainTextEdit,QPushButton,QRadioButton,QComboBox,QSpinBox,QDoubleSpinBox,QTreeView,QFileSystemModel
from PyQt5.QtCore import  QThread, pyqtSignal,QDir,QSortFilterProxyModel,QThreadPool,Qt
from ProgressWindow import ProgressWindow
from Worker import Worker
import datetime
import shutil
from Model_list import ZipSelector
import zipfile
import torch
from Model_utils import *

class Ui(QMainWindow):
    
    dic = pyqtSignal(dict)
    
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Ui/main_interface.ui', self)
        
        
        
        self.Console = self.findChild(QPlainTextEdit, 'Console')
        
        self.Browse_Database = self.findChild(QPushButton, 'Browse_Database')
        self.Browse_Model = self.findChild(QPushButton, 'Browse_Model')
        self.Delete_Database = self.findChild(QPushButton, 'Delete_Database')
        self.Delete_Model = self.findChild(QPushButton, 'Delete_Model')
        self.View_Model = self.findChild(QTreeView, 'ModelView')
        self.View_Database = self.findChild(QTreeView, 'DatabaseView')
        self.SaveName = self.findChild(QTextEdit, 'SaveName')
        
        self.Quantaziation = self.findChild(QRadioButton, 'Quantaziation')
        self.Pruning = self.findChild(QRadioButton, 'Pruning')
        self.Distilled_Knowledge = self.findChild(QRadioButton, 'Distilled_Knowledge')
        self.Process = self.findChild(QPushButton, 'Process')

        self.pruning_typeCB = self.findChild(QComboBox, 'PruningTypeCB')
        self.pruning_typeCB.addItems(['random_unstructured_pruning','random_structured_pruning',
                                      'magnitude_pruning','dynamic_pruning',
                                      'global_pruning','global_dynamic_pruning'])
        
        self.QTypeCB = self.findChild(QComboBox, 'QTypeCB')
        self.QTypeCB.addItems(['Static','Dynamic','Quantization Aware Training'])

        self.QuantizationEpochsSB = self.findChild(QSpinBox, 'QuantizationEpochsSB')

        self.BatchSizeSB = self.findChild(QSpinBox, 'BatchSizeSB')
        
        self.TrainFractionSB = self.findChild(QDoubleSpinBox, 'TrainFractionSB')
        self.ValFractionSB = self.findChild(QDoubleSpinBox, 'ValFractionSB')

        self.HalfRB = self.findChild(QRadioButton, 'Half')

        self.DesiredFormatCB = self.findChild(QComboBox, 'DesiredFormatCB')
        self.DesiredFormatCB.addItems(['int8','int16','int32','float16','float32'])
        
        self.PruningRatioSB = self.findChild(QDoubleSpinBox, 'PruningRatioSB')
        self.PruningEpochsSB = self.findChild(QSpinBox, 'PruningEpochsSB')
        self.ConvertTFLiteRB = self.findChild(QRadioButton, 'ConvertTFLite')
        self.CompressedRB = self.findChild(QRadioButton, 'Compressed')
        self.SaveUnzipedRB = self.findChild(QRadioButton, 'SaveUnziped')
        self.SaveUnzipedRB.setEnabled(False)
        
        self.Teacher_Model_Path = self.findChild(QTextEdit, 'Teacher_Model_Path')
        self.TemperatureSB = self.findChild(QSpinBox, 'TemperatureSB')
        self.AlphaSB = self.findChild(QDoubleSpinBox,'AlphaSB')
        self.DKEpochsSB = self.findChild(QSpinBox,'DKEpochsSB')
        
        self.Browse_Database.clicked.connect(self.Database_browsing)
        self.Delete_Database.clicked.connect(self.Database_delete)
        self.Browse_Model.clicked.connect(self.Model_browsing)
        self.Delete_Model.clicked.connect(self.Model_delete)
        self.Browse_Teacher_Model.clicked.connect(self.Teacher_Model_browsing)
        self.Process.clicked.connect(self.processClicked)

        self.CompressedRB.toggled.connect(self.SaveUnzipedRB.setEnabled)

        self.dirmodel = QFileSystemModel()
        self.dirmodel.setRootPath(QDir.rootPath())
        self.proxyModel = QSortFilterProxyModel()
        self.proxyModel.setFilterRegularExpression("Name")
        self.proxyModel.setSourceModel(self.dirmodel)
        
        self.View_Model.setModel(self.dirmodel)
        self.View_Model.setRootIndex(self.dirmodel.index('./Models/Input Models/'))
        self.View_Model.header().hideSection(2)
        self.View_Model.header().hideSection(3)

        self.dirdataset = QFileSystemModel()
        self.dirdataset.setRootPath(QDir.rootPath())

        self.View_Database.setModel(self.dirdataset)
        self.View_Database.setRootIndex(self.dirdataset.index('./Datasets/'))
        
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(5)
        self.worker_id_counter = 0
        self.worker_dict = {}

        if torch.cuda.is_available():
            self.device = 'cuda'
            print('Using GPU')
        else:
            self.device = 'cpu'
            print('Using CPU')
        
        self.show()
        
    def Database_browsing(self):
        
        Data_Path = QFileDialog.getOpenFileName(self,'open Directory','./Datasets/')[0]
        if Data_Path!="" : shutil.copy2(Data_Path,'./Datasets')
        
    def Database_delete(self):
        Data_Path = self.dirdataset.fileInfo(self.View_Database.selectedIndexes()[0]).absoluteFilePath()
        if Data_Path!="" : shutil. rmtree(Data_Path)

    def Model_browsing(self):
        Data_Path = QFileDialog.getOpenFileName(self, 'open File', './Models/Input Models/')[0]
        if Data_Path!="" : shutil.copy2(Data_Path,'./Models/Input Models/')
    
    def Model_delete(self):
        Data_Path = self.dirmodel.fileInfo(self.View_Model.selectedIndexes()[0]).absoluteFilePath()
        if Data_Path!="" : os.remove(Data_Path)
    
    def Teacher_Model_browsing(self):
        Data_Path = QFileDialog.getOpenFileName(self, 'open File', './Models/Teacher Models/')[0]
        if Data_Path!="" : self.Teacher_Model_Path.setText(Data_Path)
      
      
    def processClicked(self):
    
        if self.SaveName.toPlainText() == '':
            basename = "tempfile"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            save_name = "_".join([basename, suffix])
            
        elif self.SaveName.toPlainText()+'.h5' in os.listdir(os.getcwd()) or self.SaveName.toPlainText()+'.zip' in os.listdir(os.getcwd()) :
            save_name = self.SaveName.toPlainText() + "_1"
            
        else:
            save_name = self.SaveName.toPlainText()

        

    
        if len(self.View_Model.selectedIndexes())==0:
            self.Console.appendPlainText('\n###Please select a model###\n')
         
        elif self.Pruning.isChecked()==False and self.Quantaziation.isChecked()==False and self.Distilled_Knowledge.isChecked()==False:
            self.Console.appendPlainText('\n###Please select a method###\n')
            
        elif os.path.exists(self.Teacher_Model_Path.toPlainText())==False and self.Distilled_Knowledge.isChecked():
            self.Console.appendPlainText('\n###Please select a teacher model###\n')
        
        else:
            
            self.current_dataset_path = ''
            if len(self.View_Database.selectedIndexes()) != 0:
                self.current_dataset_path = self.dirdataset.fileInfo(self.View_Database.selectedIndexes()[0]).absoluteFilePath()
    
    
            self.current_model_path = self.dirmodel.fileInfo(self.View_Model.selectedIndexes()[0]).absoluteFilePath()


            if self.current_model_path.endswith('.zip'):
                with zipfile.ZipFile(self.current_model_path) as zip_file:
                    zip_file.extractall('./Models/Temp')
                self.zipselector = ZipSelector()
                self.zipselector.exec_()
                model = self.zipselector.getSelected()

                if model =="" :
                    self.Console.appendPlainText('\n###Please select a model to extract###\n')
                    return
                else :
                    if model in os.listdir('./Models/Input Models/'):
                        #si le model est deja dans le dossier on ajoute la date et l'heure au nom du model
                        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                        m1,m2 = model.split('.')
                        model = m1 + "_" + suffix + "." + m2
                    shutil.copy2(model,'./Models/Input Models/')
                    self.current_model_path = os.path.join('./Models/Input models',model)
                    self.Console.appendPlainText('\n###Model extracted###\n')
                
                for files in os.listdir('./Models/Temp'):
                    #on supprime les fichiers extraits non-choisis
                    path = os.path.join('./Models/Temp', files)
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        os.remove(path)
                    
            if self.current_model_path.endswith('.h5'):
                framework = 'tf'

            elif self.current_model_path.endswith('.pt'):
                framework = 'torch'

            worker_id = self.worker_id_counter
            self.worker_id_counter += 1
            progresswindow = self.create_progressWindow(self.current_model_path,'results/' + save_name,worker_id)

            dictio = {'console': self.Console, 'model_path': self.current_model_path,
                      'Pruning': self.Pruning.isChecked(),
                      'Quantization': self.Quantaziation.isChecked(),
                      'Knowledge_Distillation': self.Distilled_Knowledge.isChecked(), 'batch_size':self.BatchSizeSB.value(),
                      'pruning_ratio': self.PruningRatioSB.value(), 'dataset_path': self.current_dataset_path,
                      'half' : self.HalfRB.isChecked(),
                      'train_fraction': self.TrainFractionSB.value(), 'validation_fraction': self.ValFractionSB.value(),
                      'pruning_epochs': self.PruningEpochsSB.value(),
                      'pruning_type' : self.pruning_typeCB.currentText(),
                      'desired_format': self.DesiredFormatCB.currentText(),
                      'quantization_type' : self.QTypeCB.currentText(),
                      'quantization_epochs': self.QuantizationEpochsSB.value(),
                      'teacher_model_path': self.Teacher_Model_Path.toPlainText(),
                      'KD_temperature': self.TemperatureSB.value(), 'save_name': save_name,
                      'save_unziped': self.SaveUnzipedRB.isChecked(),
                      'convert_tflite': self.ConvertTFLiteRB.isChecked(),
                      'Compressed': self.CompressedRB.isChecked(), 'KD_alpha': self.AlphaSB.value(),
                      'KD_epochs': self.DKEpochsSB.value(), 'device': self.device, 'PWInstance': progresswindow,
                      'framework': framework}

            worker = Worker(dictio, worker_id)
            self.worker_dict[worker_id] = worker
            self.threadpool.start(worker)
            
            

    def updateConsole(self,string):
        self.Console.appendPlainText(string)

    def create_progressWindow(self,model_path,save_directory,worker_id):
        model_disk_space = os.stat(model_path).st_size / (1024 * 1024)
        progresswindow = ProgressWindow(model_disk_space,save_directory,worker_id)
        progresswindow.show()
        return progresswindow
    
    
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    UIWindow = Ui()
    app.exec_()
