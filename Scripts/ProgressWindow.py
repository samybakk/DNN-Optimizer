from pyqtgraph import PlotWidget,mkPen
from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QMainWindow,QLabel,QPushButton,QPlainTextEdit
from PyQt5.QtCore import  pyqtSignal, pyqtSlot
import os
import subprocess


class ProgressWindow(QMainWindow):
    
    on_pruning_epoch_end_signal = pyqtSignal(int, float)
    on_dk_epoch_end_signal = pyqtSignal(int, float)
    final_size_signal = pyqtSignal(str)
    
    update_progress_signal = pyqtSignal(str)
    undo_progress_signal = pyqtSignal()
    
    def __init__(self,model_path,save_directory):
        
        super().__init__()
        self.setWindowTitle("Progress Monitor")
        uic.loadUi('Ui/progress_window.ui', self)
        
        self.displayDirectoryPB = self.findChild(QPushButton, 'displayDirectoryPB')
        self.closePB = self.findChild(QPushButton, 'closePB')
        
        self.displayDirectoryPB.clicked.connect(self.displayDirectory)
        self.closePB.clicked.connect(lambda:self.close())
        self.closePB.clicked.connect(lambda:self.close())
        
        self.DKGraph = self.findChild(PlotWidget, 'DKGraph')
        self.pruningGraph = self.findChild(PlotWidget, 'pruningGraph')
        
        self.original_size = self.findChild(QLabel, 'original_size')
        self.original_size.setText("Original Size: "+str(round(os.stat(model_path).st_size/(1024^2),3))+" KB")
        
        self.CompleteLabel = self.findChild(QLabel, 'CompleteLabel')
        self.CompleteLabel.setHidden(True)
        
        self.final_size = self.findChild(QLabel, 'final_size')
        
        self.Progression_Console = self.findChild(QPlainTextEdit, 'Progression_Console')
        
        self.graph1_x = list()
        self.graph1_y = list()
    
        self.DKGraph.setBackground('w')
        pen = mkPen(color=(255, 0, 0))
        self.data_line1 = self.DKGraph.plot(self.graph1_x, self.graph1_y, pen=pen)
        
        self.graph2_x = list()
        self.graph2_y = list()
        
        self.pruningGraph.setBackground('w')
        pen = mkPen(color=(255, 127, 0))
        self.data_line2 = self.pruningGraph.plot(self.graph2_x, self.graph2_y, pen=pen)
        
        self.save_directory = save_directory
    
    def displayDirectory(self):
        subprocess.Popen('explorer '+os.path.abspath(self.save_directory))

    @pyqtSlot(int, float)
    def update_dk_graph_data(self, epoch, data):
    
        if epoch != 1:
            self.graph1_x.append(self.graph1_x[-1] + 1)
            self.graph1_y.append(data)
        #     logs['sparse_categorical_accuracy']
        else:
            self.graph1_x.append(1)
            self.graph1_y.append(data)
    
        self.data_line1.setData(self.graph1_x, self.graph1_y)

    @pyqtSlot(int,float)
    def update_pruning_graph_data(self,epoch, data):

        if epoch!=0:
            self.graph2_x.append(self.graph2_x[-1]+1)
            self.graph2_y.append(data) #logs['val_accuracy']
        else:
            self.graph2_x.append(0)
            self.graph2_y.append(data)

        self.data_line2.setData(self.graph2_x, self.graph2_y)
        
    @pyqtSlot(str)
    def update_final_size(self, size):
        self.CompleteLabel.setHidden(False)
        self.final_size.setText("Final Size: "+size+" KB")

    
    @pyqtSlot(str)
    def update_progress(self, text):
        self.Progression_Console.appendPlainText(text+'\n')

    def undo_progress(self):
        cursor = self.Progression_Console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.movePosition(
            QtGui.QTextCursor.PreviousBlock,
            QtGui.QTextCursor.KeepAnchor, 2)
        cursor.removeSelectedText()
        self.Progression_Console.setTextCursor(cursor)