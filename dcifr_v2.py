#!/usr/bin/env python
# coding: utf-8

# ## DCiFR v2: FairFace and DeepFace

# In[1]:


from __future__ import print_function, division # imports for deepface and fairface models
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import argparse
import glob
from tqdm import tqdm
import cv2
import face_recognition


# In[2]:


from PyQt5.QtCore import * # imports for GUI building
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (QApplication, QDialog,
                             QProgressBar, QPushButton)
from PyQt5 import *
from deepface import DeepFace 
import sys
import csv
import pandas as pd
import glob
from datetime import datetime
from pathlib import Path


# In[3]:


class QComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(QIComboBox, self).__init__(parent)
        
class MyProxyStyle(QProxyStyle): # adding design theme
    pass
    def pixelMetric(self, QStyle_PixelMetric, option=None, widget=None):

        if QStyle_PixelMetric == QStyle.PM_SmallIconSize:
            return 40
        else:
            return QProxyStyle.pixelMetric(self, QStyle_PixelMetric, option, widget)
        
class Wizard(QtWidgets.QWizard):
    
    #redefining nextId for page flow
    
    def nextId(self):
        id = self.currentId()
        if id == 1:
            if self.page1.fairface_cb.isChecked():
                return 3
            else:
                return 2
        if id == 2 or id == 3:
            return 4
        # ensures no next button - finishes on either of these based on check boxes
        if id == 4:
            return -1

    def __init__(self, parent=None):
        super(Wizard, self).__init__(parent)

        #add page 1,2
        self.page1 = Page1()
        self.setPage(1, self.page1)
        
        self.setStartId(1)
        
        #set ids for all potential pages
        #id = 2
        self.page2deep = Page2Deep()
        self.page2deep = self.setPage(2, self.page2deep)
        
        #id = 3
        self.page2fair = Page2Fair()
        self.page2fair = self.setPage(3, self.page2fair)
        
        #id = 4
        self.page3 = Page3()
        self.page3 = self.setPage(4, self.page3)
        
        self.setWindowTitle("DCiFR")
        self.setWindowIcon(QtGui.QIcon('logo.png')) # change size?
        self.setGeometry(0, 0, 800, 600)
        
        self.setStyleSheet("background-color:#F2F2F2") #background - light grey
        self.button(QWizard.CancelButton).setStyleSheet("background-color:#ADE6CA") # buttons - light green
        self.button(QWizard.FinishButton).setStyleSheet("background-color:#ADE6CA")
        self.button(QWizard.NextButton).setStyleSheet("background-color:#ADE6CA")
        
# page 1 - select desired attributes for analyzing
class Page1(QtWidgets.QWizardPage):

    def __init__(self, parent=None):
        super(Page1, self).__init__(parent)
        
        self.title_label = QLabel('Welcome to DCiFR!', self)
        self.title_label.move(200, 30)
        self.title_label.setFont(QFont('Helvetica', 20))
        self.title_label.adjustSize()

        self.subtitle_label = QLabel('Attribute Analysis Models', self)
        self.subtitle_label.move(200, 75)
        self.subtitle_label.setFont(QFont('Helvetica', 15))
        self.subtitle_label.adjustSize()
        
        title = QLabel('Please select which facial analysis model you would like to use.', self)
        title.move(120, 125)
        title.setFont(QFont('Helvetica', 10))

        #hover info 
        info = QLabel('Check the boxes that apply. Hover for more info!', self)
        info.move(50, 175)
        info.setFont(QFont('Helvetica', 8))
        myFont=QtGui.QFont()
        myFont.setItalic(True)
        info.setFont(myFont)
        
        info.adjustSize()
        title.adjustSize()
        
        #Check boxes
        
        #hbox allows these to be exclusive
        hbox = QHBoxLayout()
        
        self.deepface_cb = QCheckBox('DeepFace', self)
        self.deepface_cb.move(50, 225)
        self.fairface_cb = QCheckBox('FairFace', self)
        self.fairface_cb.move(50, 275)
        
        self.deepface_cb.adjustSize()
        self.fairface_cb.adjustSize()
        
        self.deepface_cb.setStyleSheet("background-color:#CAE6F2")
        self.fairface_cb.setStyleSheet("background-color:#CAE6F2")
        
        group = QButtonGroup(self)
        group.addButton(self.deepface_cb)
        group.addButton(self.fairface_cb)
        
        hbox.addWidget(self.deepface_cb)
        hbox.addWidget(self.fairface_cb)

        #Hovers
                        # TO-DO: CHANGE WHAT THESE SAY - HOW MUCH INFO SHOULD WE INCLUDE?? **********************
            
        self.deepface_cb.setToolTip('Check this box if you would like to analyze your image(s) using the DeepFace facial analysis machine learning model. DeepFace includes analysis of race, gender, age, and emotion.')
        self.fairface_cb.setToolTip('Check this box if you would like to analyze your image(s) using the FairFace facial analysis machine learning model. FairFace includes analysis of race and age group.')        
        
# deepface - folder upload
class Page2Deep(QtWidgets.QWizardPage):
    #upload and analyze multiple images

    # loop through images in folder to detect faces, run DeepFace, and produce CSV results 
    def detect_face_show_multiple(self, folderpath):
        filename = '/DCIFR/dcifr_Deepface_results_' + datetime.now().strftime("%Y-%m-%d-%H_%M.csv")
        
        # not sure how to create DCiFR folder to save to 
        
        docs_path = str(Path.home() / "Documents") # getting users documents path
        filename = docs_path + filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as file:
            age = ""
            race = ""
            gender = ""
            emotion = ""
            writer = csv.writer(file)
            writer.writerow(["File", "Age", "Dominant Race", "Gender", "Emotion"])
            
            print("Running analysis on " + str(len(os.listdir(folderpath))) + " pictures...")
            
            self.progress.setMaximum(len(os.listdir(folderpath)))
            self.progress.setValue(0) # this is setting the proportion of 100 that the progress
                                                                # bar will increase by each iteration
            for filename in os.listdir(folderpath):
                    
                f = os.path.join(folderpath, filename)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                img = cv2.imread(f)
                if img is None:
                    print("No file detected!")
                    continue
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    face_num = len(faces)
                    if (face_num == 1):
                        results = DeepFace.analyze(img, enforce_detection=False)
                        age = results['age']
                        race = results['dominant_race']
                        gender = results['gender']
                        emotion = results['dominant_emotion']
                        file = filename
                        writer.writerow([file, age, race, gender, emotion])
                        print("Done with file: " + str(file))
                    else:
                        file = filename
                        print("More or less than one face detected for file: " + str(file) + "!")
                        writer.writerow([file, "", "", "", ""])
                        self.progress.setValue(self.progress.value() + 1)
                        continue

                    self.progress.setValue(self.progress.value() + 1)

            print("Done!")
            
    # folder dialog
    def get_image_files(self):
        dialog = QFileDialog()
        dialog.setOption(dialog.DontUseNativeDialog, True)
        file_name = dialog.getExistingDirectory(self, "Select A Folder")
        file = os.path.join(file_name)
        
        self.detect_face_show_multiple(file)
        
    def __init__(self, parent=None):
        super(Page2Deep, self).__init__(parent)
        self.title_label = QLabel('DCiFR', self)
        self.title_label.move(300, 30)
        self.title_label.setFont(QFont('Helvetica', 20))
        self.title_label.adjustSize()
        self.title_label = QLabel('Upload Your Images Below', self)
        self.title_label.move(250, 75)
        self.title_label.setFont(QFont('Helvetica', 10))
        self.title_label.adjustSize()
        self.info = QLabel('Wait for the progress bar to fill before moving to the next page!', self)
        self.info.move(50, 250)
        self.info.setFont(QFont('Helvetica', 5))
        myFont=QtGui.QFont()
        myFont.setItalic(True)
        self.info.setFont(myFont)
        self.info.adjustSize()
        
        self.button1 = QPushButton("Select Your Folder of Images to Upload Here", self)   
        self.button1.clicked.connect(self.get_image_files)
        self.button1.setStyleSheet("background-color:#ADE6CA")
        self.button1.move(150, 150)
        
        self.progress = QProgressBar(self)
        self.progress.setGeometry(150, 200, 250, 20)
        self.progress.setStyleSheet("background-color:#CAE6F2")
        self.progress.setMaximum(100)
        
# fairface - folder upload
class Page2Fair(QtWidgets.QWizardPage):
    # returns facial analysis results using fairface given a folderpath of images 
    
    device = torch.device('cpu')

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=torch.device('cpu')))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load('fair_face_models/fairface_alldata_4race_20191111.pt', map_location=torch.device('cpu')))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    
    face_names = []
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []
    
    def fairface(self, image):
        Page2Fair.face_names.append(image)
        try:

            img = dlib.load_rgb_image(image)

            old_height, old_width, _ = img.shape

            if old_width > old_height:
                new_width, new_height = 800, int(800 * old_height / old_width)
            else:
                new_width, new_height =  int(800 * old_width / old_height), 800
            img = dlib.resize_image(img, rows=new_height, cols=new_width)

            dets = Page2Fair.cnn_face_detector(img, 1)
            num_faces = len(dets)

            if num_faces == 0:
                print("Sorry, there were no faces found in '{}'".format(image))
                pass
            # Find the 5 face landmarks we need to do the alignment.
            faces = dlib.full_object_detections()
            for detection in dets:
                rect = detection.rect
                faces.append(Page2Fair.sp(img, rect))


            image = dlib.get_face_chips(img, faces, size=300, padding = 0.25)
            image = image[0]

            trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            image = trans(image)

            image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            image = image.to(Page2Fair.device)

            outputs = Page2Fair.model_fair_7(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            race_outputs = outputs[:7]
            gender_outputs = outputs[7:9]
            age_outputs = outputs[9:18]

            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
            gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
            age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

            race_pred = np.argmax(race_score)
            gender_pred = np.argmax(gender_score)
            age_pred = np.argmax(age_score)

            Page2Fair.race_scores_fair.append(race_score)
            Page2Fair.gender_scores_fair.append(gender_score)
            Page2Fair.age_scores_fair.append(age_score)

            Page2Fair.race_preds_fair.append(race_pred)
            Page2Fair.gender_preds_fair.append(gender_pred)
            Page2Fair.age_preds_fair.append(age_pred)

            # fair 4 class
            outputs = Page2Fair.model_fair_4(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            race_outputs = outputs[:4]
            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
            race_pred = np.argmax(race_score)

            Page2Fair.race_scores_fair_4.append(race_score)
            Page2Fair.race_preds_fair_4.append(race_pred)

            print("Done with analysis!")

        except RuntimeError:
            Page2Fair.race_preds_fair.append(np.nan)
            Page2Fair.race_preds_fair_4.append(np.nan)
            Page2Fair.gender_preds_fair.append(np.nan)
            Page2Fair.age_preds_fair.append(np.nan)
            Page2Fair.race_scores_fair.append(np.nan)
            Page2Fair.race_scores_fair_4.append(np.nan)
            Page2Fair.gender_scores_fair.append(np.nan)
            Page2Fair.age_scores_fair.append(np.nan)

            print("Done with analysis - error found!")
    
    def fairface_results(self, folderpath):
        self.progress.setMaximum(len(os.listdir(folderpath)) + 1) # adding one due to format of results production
                                            # will be at 90% after for loop runs and then 100% after results are produced
        self.progress.setValue(0) # this is setting the proportion of 100 that the progress
                                                                # bar will increase by each iteration
        # run analysis
        for filename in os.listdir(folderpath):
            file = os.path.join(folderpath, filename)
            self.fairface(file)
            self.progress.setValue(self.progress.value() + 1)
            
        # produce results   
        result = pd.DataFrame([Page2Fair.face_names, Page2Fair.race_preds_fair, Page2Fair.race_preds_fair_4, 
                       Page2Fair.gender_preds_fair, Page2Fair.age_preds_fair, Page2Fair.race_scores_fair, 
                       Page2Fair.race_scores_fair_4, Page2Fair.gender_scores_fair, Page2Fair.age_scores_fair]).T

        result.columns = ['face_name_align',
                          'race_preds_fair',
                          'race_preds_fair_4',
                          'gender_preds_fair',
                          'age_preds_fair',
                          'race_scores_fair',
                          'race_scores_fair_4',
                          'gender_scores_fair',
                          'age_scores_fair']

        result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
        result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
        result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
        result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
        result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
        result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
        result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

        # race fair 4

        result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
        result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
        result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
        result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

        # gender
        result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
        result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

        # age
        result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
        result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
        result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
        result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
        result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
        result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49' 
        result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
        result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
        result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

        name = '/DCIFR/dcifr_Fairface_results_' + datetime.now().strftime("%Y-%m-%d-%H_%M.csv")
        
        docs_path = str(Path.home() / "Documents")
        filename = docs_path + name
        os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        result[['face_name_align',
                'race', 'race4',
                'gender', 'age',
                'race_scores_fair', 'race_scores_fair_4',
                'gender_scores_fair', 'age_scores_fair']].to_csv(filename, index=False)
        self.progress.setValue(self.progress.value() + 1)
        print("Done!")   
    
    # folder dialog
    def get_image_files(self):
        dialog = QFileDialog()
        dialog.setOption(dialog.DontUseNativeDialog, True)
        file_name = dialog.getExistingDirectory(self, "Select A Folder")
        file = os.path.join(file_name)
        self.fairface_results(file)
        
    def __init__(self, parent=None):
        super(Page2Fair, self).__init__(parent)
        self.title_label = QLabel('DCiFR', self)
        self.title_label.move(200, 30)
        self.title_label.setFont(QFont('Helvetica', 20))
        self.title_label.adjustSize()
        self.title_label = QLabel('Upload Your Images Below', self)
        self.title_label.move(200, 75)
        self.title_label.setFont(QFont('Helvetica', 10))
        self.title_label.adjustSize()
        self.info = QLabel('Wait for the progress bar to fill before moving to the next page!', self)
        self.info.move(150, 250)
        self.info.setFont(QFont('Helvetica', 7))
        myFont=QtGui.QFont()
        myFont.setItalic(True)
        self.info.setFont(myFont)
        self.info.adjustSize()
        
        self.button1 = QPushButton("Select Your Folder of Images to Upload Here", self)   
        self.button1.clicked.connect(self.get_image_files)
        self.button1.setStyleSheet("background-color:#ADE6CA")
        self.button1.move(150, 150)
        
        self.progress = QProgressBar(self)
        self.progress.setGeometry(150, 200, 250, 20)
        self.progress.setStyleSheet("background-color:#CAE6F2")
        self.progress.setMaximum(100)
        
# results page
class Page3(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(Page3, self).__init__(parent)
        
        self.title_label = QLabel('DCiFR - Results', self)
        self.title_label.move(250, 30)
        self.title_label.setFont(QFont('Helvetica', 20))
        self.title_label.adjustSize()

        self.title_label = QLabel('Here are the results for the image(s) you uploaded:', self)
        self.title_label.move(150, 125)
        self.title_label.setFont(QFont('Helvetica', 10))
        self.title_label.adjustSize()
        
        results_label = QLabel("Please check your DCIFR folder in your Documents \nfor a CSV results file", self)
        results_label.move(75, 200)
        myFont=QtGui.QFont('Helvetica', 14)
        myFont.setItalic(True)
        results_label.setFont(myFont)
        results_label.adjustSize()
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myStyle = MyProxyStyle('Fusion')
    app.setStyle(myStyle)
    wizard = Wizard()
    wizard.show()
    app.exec_() 

