import os
import glob, random
import subprocess
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
import os, platform
import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QLineEdit, QMessageBox, QGridLayout, QInputDialog
from PyQt5.QtGui import QIcon
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from PyQt5 import QtGui, QtWidgets, QtCore
import shutil


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


print(resource_path(''))
random.seed(10)



# class ProgressBar(QProgressBar):
    
#     def __init__(self, *args, **kwargs):
#         super(ProgressBar, self).__init__(*args, **kwargs)
#         self.setValue(0)
#         if self.minimum() != self.maximum():
#             self.timer = QTimer(self, timeout=self.onTimeout)
#             self.timer.start(random.randint(1, 3) * 1000)

#     def onTimeout(self):
#         if self.value() >= 100:
#             self.timer.stop()
#             self.timer.deleteLater()
#             del self.timer
#             return
#         self.setValue(self.value() + 1)
        
        
def automatic_brightness_and_contrast(image, clip_hist_percent=5):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
   
   
class Thread(QThread):
    _signal = pyqtSignal(int)
    def __init__(self):
        super(Thread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        with open(resource_path('logs/log.txt'),'r') as f:
            content = f.read()
            print(content.split(' ')[-1])
        noofepochs = content.split(' ')[-1]
        if int(noofepochs) <= 100:
            print('under 100')
            print((int(noofepochs)*100)-((int(noofepochs)-1)*100))
            if int(noofepochs) <= 10: 
                for i in range((int(noofepochs)*100)-((int(noofepochs)-1)*100)):
                    print(i)
                    time.sleep(2)
                    self._signal.emit(i) 
                # pbar.setValue(int(noofepochs.text())*100/100)
            elif int(noofepochs) > 10:
                for i in range((int(noofepochs)*100)//int(noofepochs)):
                    time.sleep(2)
                    self._signal.emit(i) 




unet = """


"""
# print(unet)

algorithms_list=['Unet', 'PixelHop']
# lossfunctions_list = ['Binary Cross Entropy', 'Cateogorical Cross Entropy', 'Customize']
        
class Input_tab(QWidget):
    
    def __init__(self):
        super().__init__()
        global Train, pbar, proc, rawimagepathtextbox, groundtruthtextbox, algorithms, rawimagelabel, groundtruthlabel, consoletextbox, noofclasses, noofepochs

    
    
        vlayout = QVBoxLayout()
        
        #-----------------------------------------------------INPUTS-----------------------------------------------------
        
        hlayout = QHBoxLayout()
        hlayout1 = QHBoxLayout()
        hlayout2 = QHBoxLayout()
        hlayout3 = QHBoxLayout()
        hlayout4 = QHBoxLayout()
        
        
        
        rawimagepathlabel = QLabel('Raw Images  ', self)
        rawimagepathlabel.setFont(QFont('Segoe UI', 10))
    
  
        rawimagepathtextbox = QLineEdit(self, placeholderText='Raw images folder')
        rawimagepathtextbox.setFont(QFont('Segoe UI', 8))
        rawimagepathtextbox.resize(600,30)
        
        
        rawimagepathbrowse = QPushButton('Select', self)
        rawimagepathbrowse.clicked.connect(self.on_click_rawimagefolder)
        rawimagepathbrowse.setToolTip('Select the Raw Images Folder')
        rawimagepathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        
        rawimagelabel = QLabel(self)
        groundtruthlabel = QLabel(self)
        
        
        
    
        
        algorithmslabel = QLabel('Select algorithm', self)
        algorithmslabel.setFont(QFont('Segoe UI', 10))
        
        groundtruthpathlabel = QLabel('Ground truth', self)
        groundtruthpathlabel.setFont(QFont('Segoe UI', 10))
    
      
        groundtruthtextbox = QLineEdit(self, placeholderText='Ground truth folder')
        groundtruthtextbox.setFont(QFont('Segoe UI', 8))
        groundtruthtextbox.resize(600,30)
        
        groundtruthpathbrowse = QPushButton('Select', self)
        groundtruthpathbrowse.clicked.connect(self.on_click_groundtruthfolder)
        groundtruthpathbrowse.setToolTip('Select the Ground truth Folder')
        groundtruthpathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
      
        algorithms = QComboBox()
        algorithms.setFont(QFont('Segoe UI', 8))
        algorithms.addItems(algorithms_list)
        # algorithms.addItem("2D Unet")
        # algorithms.addItem("Custom")
        algorithms.currentIndexChanged.connect(self.algo_change)
        
        classeslabel = QLabel('Number of Classes', self)
        classeslabel.setFont(QFont('Segoe UI', 10))
        
        noofclasses = QLineEdit(self)
        noofclasses.setFont(QFont('Segoe UI', 8))
        noofclasses.setFixedWidth(30)
        
        
        epochlabel =  QLabel('Number of Epochs', self)
        epochlabel.setFont(QFont('Segoe UI', 10))
        
        noofepochs = QLineEdit(self)
        noofepochs.setFont(QFont('Segoe UI', 8))
        noofepochs.setFixedWidth(30)
        
        
        # lossfunctionslabel = QLabel('Select loss function', self)
        # lossfunctionslabel.setFont(QFont('Segoe UI', 10))
        
        # lossfunctions = QComboBox()
        # lossfunctions.setFont(QFont('Segoe UI', 8))
        # lossfunctions.addItems(lossfunctions_list)
        # # lossfunctions.addItem("Binary Cross Entropy")
        # # lossfunctions.addItem("Cateogorical Cross Entropy")
        # lossfunctions.currentIndexChanged.connect(self.loss_change)
        
        
        
        
        
        consoletextbox = QPlainTextEdit(self)
        consoletextbox.setFont(QFont('Segoe UI', 10))
        consoletextbox.setFixedHeight(200)
        consoletextbox.setStyleSheet('QPlainTextEdit{ border:0; }')
        
        
        
        
        
        
        pbar = QProgressBar(self, textVisible=False)
        pbar.setValue(0)
        pbar.setFixedHeight(10)
        # pbar.setAlignment(Qt.AlignCenter)
        pbar.setStyleSheet("""QProgressBar {
border-radius: 5px;
}
QProgressBar::chunk 
{
background-color: green;
border-radius :5px;
}    
                          """)
        
        # pbar.resize(300, 100)
        
        Train = QPushButton('Run', self)
        Train.clicked.connect(self.on_click_run)
        Train.setToolTip('Train the model')
        Train.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
       
       
       
       
               
      
       
        hlayout.addWidget(rawimagepathlabel)
        hlayout.addWidget(rawimagepathtextbox)
        hlayout.addWidget(rawimagepathbrowse)
        # hlayout.addWidget(algorithms)
        
        hlayout1.addWidget(groundtruthpathlabel)
        hlayout1.addWidget(groundtruthtextbox)
        hlayout1.addWidget(groundtruthpathbrowse)
        
        hlayout4.addWidget(rawimagelabel)
        hlayout4.addWidget(groundtruthlabel)
        
        
        
        hlayout2.addWidget(classeslabel)
        hlayout2.addWidget(noofclasses,1)

        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        hlayout2.addWidget(QWidget())
        
        
        
        hlayout3.addWidget(epochlabel)
        hlayout3.addWidget(noofepochs,1)
        
     
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
        hlayout3.addWidget(QWidget())
      
        vlayout.addLayout(hlayout)
        vlayout.addLayout(hlayout1)
        vlayout.addLayout(hlayout4)
        vlayout.addWidget(QWidget())
        vlayout.addLayout(hlayout2)
        vlayout.addLayout(hlayout3)
        
        vlayout.addWidget(QWidget())
        vlayout.addWidget(algorithmslabel)
        vlayout.addWidget(algorithms)
        vlayout.addWidget(QWidget())
        # vlayout.addWidget(lossfunctionslabel)
        # vlayout.addWidget(lossfunctions)
        
        vlayout.addWidget(QWidget())
        # vlayout.addWidget(QWidget())
        vlayout.addWidget(QWidget())
        vlayout.addWidget(QWidget())
        
        vlayout.addWidget(consoletextbox)
        vlayout.addWidget(QWidget())
        vlayout.addWidget(pbar)
        vlayout.addWidget(Train)
        
        
        self.setLayout(vlayout)
        

    
    @pyqtSlot()
    def on_click_rawimagefolder(self):
        global rawimages
        rawimagepathtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        rawimagepathtextbox.setText(foldername)
        
        rawimages = glob.glob(foldername+'/*.*')
        # randomimage = random.choice(rawimages)#QPixmap(rawimages)
        # print(randomimage)
        
        
    def on_click_groundtruthfolder(self):
        global rawimages, groundtruthimages
        groundtruthtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        groundtruthimages = glob.glob(foldername+'/*.*')
        # randomgroundtruthimage = random.choice(groundtruthimages)#QPixmap(rawimages)
        # print(randomgroundtruthimage)
        groundtruthtextbox.setText(foldername)
        
        randomimage = random.randint(0, len(groundtruthimages))
        print(rawimages[randomimage])
        print(groundtruthimages[randomimage])
        
        r_img = cv2.imread(rawimages[randomimage])
        cols, rows, channel = r_img.shape
        print(r_img.shape)
        brightness= np.sum(r_img)/(255*cols*rows)
        minimum_brightness = 0.33
        ratio = brightness/minimum_brightness
        
      

        r_bright_img = cv2.convertScaleAbs(r_img, alpha = 1/ratio, beta=0)
        
        
        bytesperline = 3 * cols
        r_qimg = QImage(r_bright_img,cols,rows,bytesperline,QImage.Format_RGB888)        
        pixmap = QPixmap(r_qimg)
        pixmap4 = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        rawimagelabel.setPixmap(pixmap4)
        # rawimagelabel.setFixedSize(100,100)
        
        g_img = cv2.imread(groundtruthimages[randomimage])
        # g_img = cv2.resize(g_img, (400,400), interpolation=cv2.INTER_AREA)
        cols, rows, channel = g_img.shape
        print(g_img.shape)
        brightness= np.sum(g_img)/(255*cols*rows)
        minimum_brightness = 0.33
        ratio = brightness/minimum_brightness

        g_bright_img = cv2.convertScaleAbs(g_img, alpha = 1/ratio, beta=0) 
        
        
        g_qimg = QImage(g_bright_img,cols,rows,bytesperline,QImage.Format_RGB888) 
        
        pixmap = QPixmap(g_qimg)
        pixmap5 = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        groundtruthlabel.setPixmap(pixmap5)
        # groundtruthlabel.setFixedSize(100,100)
        
       
        
    def algo_change(self, i):
        print(algorithms.currentText())
        
        if algorithms.currentText() == "Customize":
            text, okPressed = QInputDialog.getText(self, "Algorithm","Algorithm name:", QLineEdit.Normal, "")
            if okPressed and text != '':
                print(text)
                algorithms.clear()
                algorithms_list.append(text)
                algorithms_list.sort(key = 'Customize'.__eq__)
                # algorithms_list.append(algorithms_list.pop(algorithms_list.index(len(algorithms_list)-1))) 
                algorithms.addItems(algorithms_list)
                algorithms.setCurrentText(text)

            

        
        
        
        
    # def loss_change(self, i):
    #     print(lossfunctions.currentText())
    #     if lossfunctions.currentText() == "Customize":
    #         text, okPressed = QInputDialog.getText(self, "Loss function","Loss function name:", QLineEdit.Normal, "")
    #         if okPressed and text != '':
    #             print(text)
    #             lossfunctions.clear()
    #             lossfunctions_list.append(text)
    #             lossfunctions_list.sort(key = 'Customize'.__eq__)
    #             # algorithms_list.append(algorithms_list.pop(algorithms_list.index(len(algorithms_list)-1))) 
    #             lossfunctions.addItems(lossfunctions_list)
    #             lossfunctions.setCurrentText(text)
        
        
    def on_click_run(self):
        global proc, pbar, Train, noofepochs
        print("Run")
        consoletextbox.clear()
        with open(resource_path('algorithms/3D UNET.py'),'w') as f:
            f.write(unet)
        
        with open(resource_path('logs/log.txt'),'w') as f:
            f.write('Numbar of epochs: ' + str(noofepochs.text()))
        
        
        
        self.thread = Thread()
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()
        Train.setEnabled(False)
        pbar.setValue(0)
        files = glob.glob('algorithms/*.*')
        print(files)
        for f in files:
            if f.split('\\')[1] == '3D UNET.py':
                # print(sys.executable + " \"algorithms/3D UNET.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
                proc = subprocess.Popen(sys.executable + " \"algorithms/3D UNET.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
                # proc.wait()
                # os.remove(resource_path('algorithms/3D UNET.py'))
            # elif f.split('\\')[1] == '2D UNET.py':
            #     proc = subprocess.Popen(sys.executable + " \"algorithms/2D UNET.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
            else:
                print('no script')
    
    def signal_accept(self, msg):
        
        if proc.poll() is None:
            # if int(noofepochs.text()) < 100:
            pbar.setValue(int(msg))
            # elif int(noofepochs.text()) > 100:
                # pbar.setValue(int(noofepochs.text())*100/100)
        else:
            pbar.setValue(100)
            print('Done')
            with open(resource_path('logs/outputlog.txt'),'r') as f:
                score = f.read()
            
            consoletextbox.appendPlainText('Results:\n')    
            consoletextbox.appendPlainText('Loss: ' + score.split('\n')[0].split(' ')[2])
            consoletextbox.appendPlainText('IOU Score: ' + score.split(' ')[-1])
                    
            Train.setEnabled(True)
            self.thread.terminate()
        

    
        
class Test_tab(QWidget):
    
    def __init__(self):
        super().__init__()
        global algorithms, lossfunctions, testimagespathtextbox, modelpathtextbox, savepredictedtextbox
        
        
        hlayout2 = QHBoxLayout()
        hlayout3 = QHBoxLayout()
        hlayout4 = QHBoxLayout()
        vlayout4 = QVBoxLayout()
        
         
        testimageslabel = QLabel('Test Images  ', self)
        testimageslabel.setFont(QFont('Segoe UI', 10))
    
  
        testimagespathtextbox = QLineEdit(self, placeholderText='Test images folder')
        testimagespathtextbox.setFont(QFont('Segoe UI', 8))
        testimagespathtextbox.resize(600,30)
        
        
        testimagespathbrowse = QPushButton('Select', self)
        testimagespathbrowse.clicked.connect(self.test_image)
        testimagespathbrowse.setToolTip('Select the Test Images Folder')
        testimagespathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
  
        
        modellabel = QLabel('  Model File  ', self)
        modellabel.setFont(QFont('Segoe UI', 10))
    
  
        modelpathtextbox = QLineEdit(self, placeholderText='Model (.h5) file (if available)')
        modelpathtextbox.setFont(QFont('Segoe UI', 8))
        modelpathtextbox.resize(600,30)
        
        
        modelfilebrowse = QPushButton('Select', self)
        modelfilebrowse.clicked.connect(self.model_file)
        modelfilebrowse.setToolTip('Select the Model File')
        modelfilebrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        
        savepredictedlabel = QLabel('Save Predicted Images  ', self)
        savepredictedlabel.setFont(QFont('Segoe UI', 10))
    
  
        savepredictedtextbox = QLineEdit(self, placeholderText='Save Predicted Images folder')
        savepredictedtextbox.setFont(QFont('Segoe UI', 8))
        savepredictedtextbox.resize(600,30)
        
        
        savepredictedbrowse = QPushButton('Select', self)
        savepredictedbrowse.clicked.connect(self.savepredicted_image)
        savepredictedbrowse.setToolTip('Select the Save Predicted Images Folder')
        savepredictedbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        
        
        Test = QPushButton('Test', self)
        Test.clicked.connect(self.testing)
        Test.setToolTip('Test the model')
        Test.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
       
       
       
        hlayout2.addWidget(testimageslabel)
        hlayout2.addWidget(testimagespathtextbox)
        hlayout2.addWidget(testimagespathbrowse)
        
        hlayout3.addWidget(modellabel)
        hlayout3.addWidget(modelpathtextbox)
        hlayout3.addWidget(modelfilebrowse)
               
        hlayout4.addWidget(savepredictedlabel)
        hlayout4.addWidget(savepredictedtextbox)
        hlayout4.addWidget(savepredictedbrowse)
      
        
        vlayout4.addLayout(hlayout2)
        vlayout4.addLayout(hlayout3)
        vlayout4.addLayout(hlayout4)
        
        
        
        vlayout4.addWidget(QWidget())
        vlayout4.addWidget(QWidget())
        vlayout4.addWidget(QWidget())
        
        
        vlayout4.addWidget(Test)
        
        
        self.setLayout(vlayout4)
        
        
        
       
    @pyqtSlot()
    def savepredicted_image(self):
        global savepredictedtextbox
        print("select predicted images folder")
        savepredictedtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        savepredictedtextbox.setText(foldername)
        
    def test_image(self):
        global testimagespathtextbox
        print("select test images folder")
        testimagespathtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        testimagespathtextbox.setText(foldername)
    
    def model_file(self):
        global modelpathtextbox
        print("select model file")
        modelpathtextbox.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Select File')
        modelpathtextbox.setText(filename[0])

    
    def testing(self):
        global testimagespathtextbox,modelpathtextbox, savepredictedtextbox

        
        unet_test = """

from unittest.mock import patch
import tensorflow as tf
import keras
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import MeanIoU
from PIL import Image
import cv2
import random
import tifffile
from keras.utils.np_utils import normalize
from keras.models import load_model
from tifffile import imsave
import re
import os, sys
from skimage import color
import os,re, cv2, numpy as np, patchify, imgaug
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from skimage import color, io
from PIL import Image as im
import ipyplot
from patchify import patchify, unpatchify
import getopt

from keras.losses import binary_crossentropy
beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1


IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_DEPTH = 128
IMG_CHANNELS = 3
step_size = 128


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hd: r: m: s:",["rfile = ","mfile = ", "sfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -r <rawimages> -m <modelfile> -s <savefiles>(getopterror)")
        sys.exit(2)
        
        
    for opt, arg in opts:
        if opt == "-r":
            path = arg
        elif opt in ("-m", "--mfile"):
            mpath = arg
        elif opt in ("-s", "--sfile"):
            spath = arg
        else:
            print("Pythonfile.py -r <rawimages> -m <modelfile> -s <savefiles>")
            sys.exit()
        

    def dice_coefficient(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f*y_pred_f)
        return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


    def dice_coefficient_loss(y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)


    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_DEPTH = 128
    IMG_CHANNELS = 3
    step_size = 128

        

        
    dependencies = {
        'dice_coefficient': dice_coefficient,
        'dice_coefficient_loss': dice_coefficient_loss,
        
    }


    my_model = load_model(mpath, custom_objects = dependencies)

        
    path = path

    print(path)
    dimensions = IMG_HEIGHT


    def PIL2array(img):
        img = cv2.resize(img, (dimensions, dimensions))
        img = np.array(img, dtype=np.uint8)
        return(img)



    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)


    dirlist = sorted_alphanumeric(os.listdir(path))
    FRAMES = []
    FIRST_SIZE = None

    #   if allImages:
    for fn in dirlist:
        img = cv2.imread((os.path.join(path, fn)))
        if FIRST_SIZE is None:
            FIRST_SIZE = img.size
        if img.size == FIRST_SIZE:
            FRAMES.append(PIL2array(img))
            # #Horizontal Flip
            input_img = PIL2array(img)
            hflip = iaa.Fliplr(p=1.0)
            input_hf = hflip.augment_image(input_img)
            # _ = color.rgb2gray(input_hf)
            FRAMES.append(np.array(input_hf, np.uint8))
            #imgLabels.append("Horiz Flip")

            # Vertical Flip
            vflip = iaa.Flipud(p=1.0)
            input_vf = vflip.augment_image(input_img)
            # _ = color.rgb2gray(input_vf)
            FRAMES.append(np.array(input_vf, np.uint8))
            #imgLabels.append("Vertical Flip")

            # Rotation
            rot1 = iaa.Affine(rotate=(-90, 90))
            input_rot1 = rot1.augment_image(input_img)
            # _ = color.rgb2gray(input_rot1)
            FRAMES.append(np.array(input_rot1, np.uint8))
            #imgLabels.append("Rotation")

            ## Gaussian
            transform = iaa.AdditiveGaussianNoise(scale=(0, 0.009125 * 255))
            transformedImg = transform.augment_image(image=input_img)
            FRAMES.append(np.array(transformedImg, np.uint8))
            #imgLabels.append("Gaussian Noise")

            ## Laplace Noise

            laplace = iaa.AdditiveLaplaceNoise(scale=(0, 0.0008 * 255))
            laplaceImg = laplace.augment_image(image=input_img)
            FRAMES.append(np.array(laplaceImg, np.uint8))
            #imgLabels.append("Laplace Noise")

            ## Poisson Noise
            poisson = iaa.AdditivePoissonNoise((0, 3))
            poissonImg = poisson.augment_image(image=input_img)
            FRAMES.append(np.array(poissonImg, np.uint8))
            #imgLabels.append("Poisson Noise")

            ## Image Shearing
            shear = iaa.Affine(shear=(-40, 40))
            input_shear = shear.augment_image(input_img)
            # _ = color.rgb2gray(input_shear)
            FRAMES.append(np.array(input_shear, np.uint8))
            #imgLabels.append("Shearing Image")
            #print(type(imgList))
            print(fn, ": has been added")
        else:
            print("Discard:", fn, img.size, "<>", FIRST_SIZE)




    FRAMES = np.array(FRAMES)

    def rgb2grayPersonal(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    binary = True#input("Is this for binary segmentation? Enter True or False >>> ")

    if binary:
        gray = rgb2grayPersonal(FRAMES)
        FRAMES = gray



    temp = FRAMES
    img = temp

    print(img.shape)
    img_patches = patchify(img, (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), step=step_size)

    print(img_patches.shape)






    predicted_patches = []
    # cnt = 0
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            for k in range(img_patches.shape[2]):
                single_patch = img_patches[i,j,k, :,:,:]
                single_patch_3ch = np.stack((single_patch,)*3, axis=-1) #make rgb
                single_patch_3ch = normalize(single_patch_3ch, axis=1)
                single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0) #expand dimensions
                single_patch_prediction = my_model.predict(single_patch_3ch_input)
                single_patch_prediction_argmax = np.argmax(single_patch_prediction, axis=4)[0,:,:,:]
                predicted_patches.append(single_patch_prediction_argmax)





    #Convert list to numpy array
    predicted_patches = np.array(predicted_patches)



    #Reshape to the shape we had after patchifying
    predicted_patches_reshaped = np.reshape(predicted_patches, 
                                            (img_patches.shape[0], img_patches.shape[1], img_patches.shape[2],
                                             img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]) )





    # Repach individual patches into the orginal volume shape
    reconstructed_image = unpatchify(predicted_patches_reshaped, img.shape)


    #Convert to uint8 so we can open image in most image viewing software packages
    reconstructed_image=reconstructed_image.astype(np.uint8)

    print('now saving')
    
    #Now save it as segmented volume.

    imsave(spath + "/segmented.tif", reconstructed_image)

    print('done')

if __name__ == "__main__":
      
      main(sys.argv[1:])
      

            
            
            
        
        """
        
        
        with open(resource_path('algorithms/3D UNET_test.py'),'w') as f:
            f.write(unet_test)
        
        # with open(resource_path('log.txt'),'w') as f:
        #     f.write('Numbar of epochs: ' + str(noofepochs.text()))
        
        
        
        # self.thread = Thread()
        # self.thread._signal.connect(self.signal_accept)
        # self.thread.start()
        # Train.setEnabled(False)
        # # pbar.setValue(0)
        print(sys.executable + " \"algorithms/3D UNET_test.py\" " +"-r \""+ testimagespathtextbox.text() +"\""+ " -m \""+ modelpathtextbox.text()+"\""+" -s \""+savepredictedtextbox.text()+"\"")
        
        
        
        files = glob.glob('algorithms/*.*')
        print(files)
        for f in files:
            if f.split('\\')[1] == '3D UNET_test.py':
                # print(sys.executable + " \"algorithms/3D UNET.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
                proc = subprocess.Popen(sys.executable + " \"algorithms/3D UNET_test.py\" " +"-r \""+ testimagespathtextbox.text() +"\""+ " -m \""+ modelpathtextbox.text()+"\""+" -s \""+savepredictedtextbox.text()+"\"")
                # proc.wait()
                # os.remove(resource_path('algorithms/3D UNET_test.py'))
            # elif f.split('\\')[1] == '2D UNET_test.py':
            #     proc = subprocess.Popen(sys.executable + " \"algorithms/2D UNET_test.py\" -r \""+rawimagepathtextbox.text()+"\" -g \""+groundtruthtextbox.text()+"\" -c \""+noofclasses.text()+"\" -e \""+noofepochs.text()+"\"")
            else:
                print('no script')
        
        
class About_tab(QWidget):
    def __init__(self):
        super().__init__()
        aboutLayout = QGridLayout()
        logoPic = QPixmap("utd logo circular.png")
        label = QLabel(self)
        label.setPixmap(logoPic)
        # label.setText("Created by Aayan Rahat and Vinay Kadam, Researchers at UTD and the Ding Incubator")
        label.setAlignment(QtCore.Qt.AlignCenter)
        aboutLayout.addWidget(label)
        message = QLabel(
            'Reconstruct-Net: Created by Aayan Rahat and Vinay Kadam, Researchers at UTD and the Ding Incubator')
        message.setAlignment(QtCore.Qt.AlignCenter)
        aboutLayout.addWidget(message)
        message2 = QLabel('For all code and latest revisions, please visit our Github')
        message2.setAlignment(QtCore.Qt.AlignCenter)
        aboutLayout.addWidget(message2)
        self.setLayout(aboutLayout)
        
class Export_tab(QWidget):
    
    def __init__(self):
        super().__init__()
        global rawimagepathtextbox, groundtruthtextbox, algorithms, lossfunctions, segmentedimagespathtextbox
        
        hlayout4 = QHBoxLayout()
        vlayout5 = QVBoxLayout()
        
        Segmentedimageslabel = QLabel('Segmented Images ', self)
        Segmentedimageslabel.setFont(QFont('Segoe UI', 10))
    
  
        segmentedimagespathtextbox = QLineEdit(self, placeholderText='Segmented images folder')
        segmentedimagespathtextbox.setFont(QFont('Segoe UI', 8))
        segmentedimagespathtextbox.resize(600,30)
        
        
        segmentedimagespathbrowse = QPushButton('Select', self)
        segmentedimagespathbrowse.clicked.connect(self.segmented_images)
        segmentedimagespathbrowse.setToolTip('Select the Segmented Images Folder')
        segmentedimagespathbrowse.setStyleSheet('QPushButton {  border-color: white; border-width: 0px; padding: 10px; border-style: outset; border-radius: 6px; background-color : #3375ec; font-weight: regular; font-size: 8pt; font-family: Seoge UI; color: white; }')#setStyleSheet('QPushButton {  background-color : #92374d; color: white; }')
        

        
        
        
        
  
        export = QPushButton('Export 3D', self)
        export.clicked.connect(self.export)
        export.setToolTip('Export the 3D model')
        export.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
        # export4 = QPushButton('Export 4D', self)
        # export4.clicked.connect(self.export)
        # export4.setToolTip('Export the 4D model')
        # export4.setStyleSheet('QPushButton {  border-color: black; border-width: 0.2px; padding: 14px; border-style: outset; border-radius: 20px; background-color : lightgray; font-weight: regular; font-size: 12pt; font-family: Seoge UI; color: black; } }')
       
        hlayout4.addWidget(Segmentedimageslabel)
        hlayout4.addWidget(segmentedimagespathtextbox)
        hlayout4.addWidget(segmentedimagespathbrowse)
       
       
        vlayout5.addLayout(hlayout4)
       
        vlayout5.addWidget(QWidget())
        vlayout5.addWidget(QWidget())
        vlayout5.addWidget(QWidget())
        vlayout5.addWidget(export)
        
        
        self.setLayout(vlayout5)
  
    @pyqtSlot()
    def segmented_images(self):
        global segmentedimagespathtextbox
        print('selected segmented images folder')
        segmentedimagespathtextbox.setText('')
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        segmentedimagespathtextbox.setText(foldername)
    
    

        
        
    def export(self):
        print("export 3D")
        
        export_vti = """
import os
import sys
import tifftools
import vtk
import numpy
import cv2
import re
import argparse
import pyvista as pv
from os import system, name
import getopt


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hd: s: ",["sfile = "])
    except getopt.GetoptError:
        print("Pythonfile.py -s <segmentedimages>")
        sys.exit(2)
    
      
    for opt, arg in opts:
        if opt == "-s":
            path = arg
     
        else:
            print("Pythonfile.py -s <segmentedimages>")
            sys.exit()
        
    IMG_HEIGHT = 64

    path = path

    dimensions = IMG_HEIGHT 

    finalDirectory = resource_path('')
    finalName = "unity"
    extension = "vti"
    finalDirectory = finalDirectory + "" + finalName

   

    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)

    dirlist = sorted_alphanumeric(os.listdir(path))

    def PIL2array(img):
        img = cv2.resize(img, (dimensions, dimensions))
  
        
        return numpy.array(img, numpy.uint8)

    FRAMES = []
    FIRST_SIZE = None

    for fn in dirlist:
        img = cv2.imread((os.path.join(path, fn)))
        if FIRST_SIZE is None:
            FIRST_SIZE = img.size
        if img.size == FIRST_SIZE:
            FRAMES.append(PIL2array(img))
        else:
            print("Discard:", fn, img.size, "<>", FIRST_SIZE)



    binary = True 

    if bool == "True":
        FRAMES[FRAMES > 0] = 1


    Stack = numpy.dstack(FRAMES)
    data = pv.wrap(Stack)

    j = IMG_HEIGHT 
    i = 0.3


    finDir = finalDirectory
    data.spacing = (j, j, i )
    finDir += ( "." + extension )
    data.save(finDir)
    print("Your file has been saved to the following location {}".format(finDir))
    
if __name__ == "__main__":
      main(sys.argv[1:])
        
        """

        with open(resource_path('bin/export_vti.py'),'w') as f:
            f.write(export_vti)
        
        # with open(resource_path('log.txt'),'w') as f:
        #     f.write('Numbar of epochs: ' + str(noofepochs.text()))
        
        
        
        # self.thread = Thread()
        # self.thread._signal.connect(self.signal_accept)
        # self.thread.start()
        # Train.setEnabled(False)
        # # pbar.setValue(0)
        print(sys.executable + " \""+resource_path('')+"bin/export_vti.py\" " +"-s \""+ segmentedimagespathtextbox.text() +"\"")
        
        
        
        files = glob.glob(resource_path('bin/*.*'))
        print(files)
        for f in files:
            if f.split('\\')[-1] == 'export_vti.py':
                print("its there")
                proc = subprocess.Popen(sys.executable + " \""+resource_path('')+"bin/export_vti.py\" " +"-s \""+ segmentedimagespathtextbox.text() +"\"")
                proc.wait()
                os.remove(resource_path('')+"bin/export_vti.py")
            else:
                print('no script')
       
       
        docker_script = """

import os
import sys
import subprocess
import time


userprofile = os.environ['USERPROFILE']

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

if os.path.exists(resource_path('Slicer_docker.dll')):
    print('slicerdocker already install')
else:
    #install docker image and create container
    install = subprocess.Popen("docker run -d --name slicernotes -p 8888:8888 -p49053:49053 -v {}:/home/sliceruser/work --rm -ti slicer/slicer-notebook:latest".format(userprofile), shell = False)
    install.wait()
    time.sleep(2)
    
    #install jupyter nbconvert into docker
    install_nbconvert = subprocess.Popen('winpty docker exec -it slicernotes bash -c \"./Slicer-4.11.0-2020-05-11-linux-amd64/Slicer --launch PythonSlicer -m pip install jupyter nbconvert\"')
    install_nbconvert.wait()

    time.sleep(2)

    #make directory inside docker
    make_obj_directory = subprocess.Popen('winpty docker exec -it slicernotes bash -c \"mkdir -p obj\"')
    make_obj_directory.wait()

    time.sleep(2)
    with open(resource_path('Slicer_docker.dll'), 'w') as f:
        f.write("Slicer for Ding's Lab")

# copy files to docker
vti_filename = resource_path("unity.vti")
vtifile = "unity.vti"
jupyter_filename = resource_path("main.ipynb")
jupyterfile = "main.ipynb"


vti_copy = subprocess.Popen("docker cp {0} slicernotes:./home/sliceruser/{1}".format(vti_filename,vtifile))
vti_copy.wait()

jupyter_copy = subprocess.Popen("docker cp {0} slicernotes:./home/sliceruser/{1}".format(jupyter_filename, jupyterfile))
jupyter_copy.wait()

time.sleep(2)

#run jupyter command inside docker
run_jupyter = subprocess.Popen('winpty docker exec -it slicernotes bash -c \"jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute main.ipynb\"')
run_jupyter.wait()

# time.sleep(`0`)

# copy files from docker
obj_filename= "Segmentation.obj"
obj_copy_docker = subprocess.Popen("docker cp slicernotes:./home/sliceruser/obj/{0} {1}".format(obj_filename, obj_filename))
obj_copy_docker.wait()




mtl_filename= "Segmentation.mtl"
mtl_copy_docker = subprocess.Popen("docker cp slicernotes:./home/sliceruser/obj/{0} {1}".format(mtl_filename, mtl_filename))

mtl_copy_docker.wait()

"""
        with open(resource_path('bin/export_model.py'),'w') as f:
            f.write(docker_script)

   
        print(sys.executable + " \""+resource_path('')+"bin/export_model.py\"")
        
        files = glob.glob(resource_path('bin/*.*'))
        print(files)
        for f in files:
            print(f.split('\\')[-1])
            if f.split('\\')[-1] == 'export_model.py':
                print("its there")
                try:
                    shutil.move(resource_path('export files/vti/unity.vti'),resource_path('bin/unity.vti'))
                except:
                    print('no vti found')
                    
                proc = subprocess.Popen(sys.executable + " \""+resource_path('')+"bin/export_model.py\"" )
                proc.wait()
                os.remove(resource_path('')+"bin/export_model.py")
                # os.remove(resource_path('')+"bin/unity.vti")
                try:
                    shutil.move(resource_path('Segmentation.obj'),resource_path('export files/models/Segmentation.obj'))
                    shutil.move(resource_path('Segmentation.mtl'),resource_path('export files/models/Segmentation.mtl'))
                except:
                    print('no files found')
                
            else:
                print('no script')
       
    


        
class TabBar(QTabBar):
    def tabSizeHint(self, index):
        s = QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)
            painter.save()

            s = opt.rect.size()
            s.transpose()
            r = QtCore.QRect(QtCore.QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r

            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(90)
            painter.translate(-c)
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt)
            painter.restore()

class VerticalTabWidget(QTabWidget):
    def __init__(self, *args, **kwargs):
        QTabWidget.__init__(self, *args, **kwargs)
        self.setTabBar(TabBar())
        self.setTabPosition(QtWidgets.QTabWidget.West)       
        
        
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        
        
        self.setWindowTitle('Segmentation Models')
        self.setFixedWidth(1024)
        self.setFixedHeight(960)
        self.setWindowIcon(QIcon(str(resource_path('logo.png'))))
        # self.setGeometry(self.top, self.left, self.width, self.height)
        frameGm = self.frameGeometry()
        screen = PyQt5.QtWidgets.QApplication.desktop().screenNumber(PyQt5.QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = PyQt5.QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
        layout = QGridLayout()
        self.setLayout(layout)
        # label1 = QLabel("Widget in Tab 1.")
        # label2 = QLabel("Widget in Tab 2.")
        tabwidget = VerticalTabWidget()
        tabwidget.update()
        tabwidget.setCursor(QCursor(Qt.PointingHandCursor))
        
        # tabwidget.setTabShape(QTabWidget.Rounded)
        # tabwidget.setTabPosition(QTabWidget.West)
        tabwidget.addTab(About_tab(), "About")
        tabwidget.addTab(Input_tab(), "Train")
        tabwidget.addTab(Test_tab(), "Test")
        #tabwidget.addTab(Export_tab(), "Export")
        # tabwidget.setDocumentMode(True)
        tabwidget.setStyleSheet('''QTabBar::tab { height: 100px; width: 50px; font-size: 10pt; font-family: Segoe UI; font-weight: Bold;}
                                ''')
 
        layout.addWidget(tabwidget, 0, 0)
        
        
        self.show()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainApp()
   
    sys.exit(app.exec_())
