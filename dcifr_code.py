#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import *
from deepface import DeepFace 
import sys
import os
import csv
import cv2
import pandas as pd


# In[ ]:


# TO DO:
    # threshold?
    # work through possible errors
    # aesthetics/design
    
    # add date and time to results csv

#threshold value is collected from slider, but no deepface yet
threshold = 50

#list of characteristics to analyze - page 1 check boxes
analyze_list = []

class QComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(QIComboBox, self).__init__(parent)
        
class Wizard(QtWidgets.QWizard):
    
    #redefining nextId for page flow
    def nextId(self):
        id = self.currentId()
        if id == 2:
            if self.page2.batch_cb.isChecked():
                return 5
            else:
                return 3
        if id == 1:
            return 2
        if id == 3:
            return 4
        if id == 5:
            return 6
        # ensures no next button - finishes on either of these based on check boxes
        if id == 6 or id == 4:
            return -1

    def __init__(self, parent=None):
        super(Wizard, self).__init__(parent)

        #add page 1,2
        self.page1 = Page1()
        self.setPage(1, self.page1)
        
        self.page2 = Page2()
        self.setPage(2, self.page2)
        
        self.setStartId(1)
        
        #set ids for all potential pages
        #id = 3
        self.page3single = Page3Single()
        self.page3singleid = self.setPage(3, self.page3single)
        
        #id = 4
        self.page4single = Page4Single()
        self.page4single.setFinalPage(True)
        self.page4singleid = self.setPage(4, self.page4single)
        
        #id = 5
        self.page3batch = Page3Batch()
        self.page3batchid = self.setPage(5, self.page3batch)
        
        #id = 6
        self.page4batch = Page4Batch()
        self.page4batch.setFinalPage(True)
        self.page4batchid = self.setPage(6, self.page4batch)
        
        self.setWindowTitle("DCiFR")
        self.setGeometry(0, 0, 800, 600)
    
# page 1 - select desired attributes for analyzing
class Page1(QtWidgets.QWizardPage):
    # doesn't account for multiple checks
    def btnstate(self,b):
        if b.isChecked() == True:
            add = b.text().lower()
            if add not in analyze_list:
                analyze_list.append(add)
        elif b.isChecked() == False:
            if b.text().lower() in analyze_list:
                analyze_list.remove(b.text().lower())
    # for threshold         
    def ValueContrast(self, value):
        threshold = value
        
    def __init__(self, parent=None):
        super(Page1, self).__init__(parent)
        
        self.title_label = QLabel('Welcome to DCiFR!', self)
        self.title_label.move(50, 30)
        self.title_label.setFont(QFont('Arial', 20))
        self.title_label.adjustSize()

        self.title_label = QLabel('Attributes', self)
        self.title_label.move(100, 75)
        self.title_label.setFont(QFont('Arial', 15))
        self.title_label.adjustSize()

        #hover info 
        info = QLabel('Check the boxes that apply. Hover for more info!', self)
        info.move(50, 125)
        info.setFont(QFont('Arial', 10))
        info.adjustSize()
        myFont=QtGui.QFont()
        myFont.setItalic(True)
        info.setFont(myFont)
        info.adjustSize()
        
        #Check boxes
        self.age_cb = QCheckBox('Age', self)
        self.age_cb.move(50, 150)
        self.race_cb = QCheckBox('Race', self)
        self.race_cb.move(50, 200)
        self.age_cb.adjustSize()
        self.race_cb.adjustSize()
        self.gender_cb = QCheckBox('Gender', self)
        self.gender_cb.move(50, 250)
        self.emotion_cb = QCheckBox('Emotion', self)
        self.emotion_cb.move(50, 300)
        self.gender_cb.adjustSize()
        self.emotion_cb.adjustSize()
        
        # connecting checkbox changes to page actions
        self.age_cb.stateChanged.connect(lambda:self.btnstate(self.age_cb))
        self.race_cb.stateChanged.connect(lambda:self.btnstate(self.race_cb))
        self.gender_cb.stateChanged.connect(lambda:self.btnstate(self.gender_cb))
        self.emotion_cb.stateChanged.connect(lambda:self.btnstate(self.emotion_cb))

        #Adding slider for race attribute
        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setRange(0, 100)
        self.sld.move(500, 240)
        self.sld.setTickInterval(10)
        label = QLabel('Please select your threshold value \n(only applicable if you select the race attribute)', self)
        label.move(300, 150)
        minlabel = QLabel('0', self)
        minlabel.move(500, 260)
        maxlabel = QLabel('100', self)
        maxlabel.move(570, 260)
        q1label = QLabel('25', self)
        q1label.move(510, 260)
        medlabel = QLabel('50', self)
        medlabel.move(530, 260)
        q3label = QLabel('75', self)
        q3label.move(550, 260)
        minlabel.setFont(QFont('Arial', 5))
        maxlabel.setFont(QFont('Arial', 5))
        q1label.setFont(QFont('Arial', 5))
        medlabel.setFont(QFont('Arial', 5))
        q3label.setFont(QFont('Arial', 5))
        minlabel.adjustSize()
        maxlabel.adjustSize()
        q1label.adjustSize()
        medlabel.adjustSize()
        q3label.adjustSize()
        
        # tool tip
        self.sld.setToolTip('This is a slider for the threshold of the race attribute.')
        self.sld.valueChanged[int].connect(self.ValueContrast)
        
        #Hovers
        self.age_cb.setToolTip('Check this box if you would like to analyze the age of the subject in your image(s).')
        self.race_cb.setToolTip('Check this box if you would like to analyze the race of the subject in your image(s).')
        self.gender_cb.setToolTip('Check this box if you would like to analyze the gender of the subject in your image(s).')
        self.emotion_cb.setToolTip('Check this box if you would like to analyze the emotion of the subject in your image(s).')
        
#page 2 - choose single or batch mode        
class Page2(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(Page2, self).__init__(parent)
        self.title_label = QLabel('DCiFR', self)
        self.title_label.move(50, 30)
        self.title_label.setFont(QFont('Arial', 20))
        self.title_label.adjustSize()
        
        self.title_label = QLabel('Single or Batch Mode', self)
        self.title_label.move(100, 75)
        self.title_label.setFont(QFont('Arial', 15))
        self.title_label.adjustSize()
        #hover info 
        info = QLabel('Check the box that applies. Hover for more info!', self)
        info.move(50, 125)
        info.setFont(QFont('Arial', 10))
        info.adjustSize()
        myFont=QtGui.QFont()
        myFont.setItalic(True)
        info.setFont(myFont)
        info.adjustSize()
        
        #Check boxes - hbox allows these to be exclusive
        hbox = QHBoxLayout()
    
        self.single_cb = QCheckBox('Single Image', self)
        self.single_cb.move(50, 150)
        self.batch_cb = QCheckBox('Batch Mode', self)
        self.batch_cb.move(50, 200)
        self.single_cb.adjustSize()
        self.batch_cb.adjustSize()

        group = QButtonGroup(self)
        group.addButton(self.single_cb)
        group.addButton(self.batch_cb)
        
        hbox.addWidget(self.single_cb)
        hbox.addWidget(self.batch_cb)
        
        #Hovers
        self.single_cb.setToolTip('Check this box if you would like to analyze demograhpics for a single image.')
        self.batch_cb.setToolTip('Check this box if you would like to analyze demograhpics for more than one image.')

# single mode - file upload
class Page3Single(QtWidgets.QWizardPage):
    # upload and analyze a single image
    
    # detect faces, run DeepFace, and produce CSV results
    def detect_face_show(self, fpath):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(fpath)
        if img is None:
            print("No file deteced!")
            return(0)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_num = len(faces)
            if (face_num == 1):
                results = DeepFace.analyze(img, analyze_list, enforce_detection=False)
                # results CSV file
                age = ""
                race = ""
                gender = ""
                emotion = ""
                for item in analyze_list:
                    if item == "age":
                        age = results['age']
                    elif item == "race":
                        race = results['dominant_race']
                    elif item == "gender":
                        gender = results['gender']
                    elif item == "emotion":
                        emotion = results['dominant_emotion']
                with open('dcifr_results.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Age", " Dominant Race", " Gender", " Emotion"])
                    writer.writerow([age, race, gender, emotion])
                print("Done!")
            else:
                print("More or less than one face detected!")
                return(0)
    
    # file dialog
    def get_image_file(self):
            dialog = QFileDialog()
            file_name = dialog.getOpenFileName(self, 'Open image')
            file = os.path.join(file_name[0])
            print(file)
            self.detect_face_show(file)
                
    def __init__(self, parent=None):
        super(Page3Single, self).__init__(parent)
        self.title_label = QLabel('DCiFR', self)
        self.title_label.move(50, 30)
        self.title_label.setFont(QFont('Arial', 20))
        self.title_label.adjustSize()
        self.title_label = QLabel('Upload Your Image Below', self)
        self.title_label.move(100, 75)
        self.title_label.setFont(QFont('Arial', 10))
        self.title_label.adjustSize()
        
        self.button1 = QPushButton("Select An Image to Upload Here", self)  
        self.button1.clicked.connect(self.get_image_file) 
        self.button1.move(50, 150)
        
# batch mode - folder upload
class Page3Batch(QtWidgets.QWizardPage):
    #upload and analyze multiple images

    # loop through images in folder to detect faces, run DeepFace, and produce CSV results 
    def detect_face_show_multiple(self, folderpath):
        with open('dcifr_results.csv', 'w', newline='') as file:
            age = ""
            race = ""
            gender = ""
            emotion = ""
            writer = csv.writer(file)
            writer.writerow(["File", "Age", "Dominant Race", "Gender", "Emotion"])
            
            print("Running analysis on " + str(len(os.listdir(folderpath))) + " pictures...")
            
            for filename in os.listdir(folderpath):
                f = os.path.join(folderpath, filename) # slashes are the wrong way??
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                img = cv2.imread(f)
                if img is None:
                    print("No file detected!")
                    return(0)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    face_num = len(faces)
                    if (face_num == 1):
                        results = DeepFace.analyze(img, analyze_list, enforce_detection=False)
                        #print(results)
                        for item in analyze_list:
                            if item == "age":
                                age = results['age']
                            elif item == "race":
                                race = results['dominant_race']
                            elif item == "gender":
                                gender = results['gender']
                            elif item == "emotion":
                                emotion = results['dominant_emotion']
                        file = filename
                        writer.writerow([file, age, race, gender, emotion])
                        print("Done with file: " + str(file))
                    else:
                        file = filename
                        print("More or less than one face detected for file: " + str(file) + "!")
                        writer.writerow([file, "", "", "", ""])
                        return(0)
            print("Done!")

    # folder dialog
    def get_image_files(self):
        dialog = QFileDialog()
        dialog.setOption(dialog.DontUseNativeDialog, True)
        file_name = dialog.getExistingDirectory(self, "Select A Folder")
        file = os.path.join(file_name)
        self.detect_face_show_multiple(file)
        
    def __init__(self, parent=None):
        super(Page3Batch, self).__init__(parent)
        self.title_label = QLabel('DCiFR', self)
        self.title_label.move(50, 30)
        self.title_label.setFont(QFont('Arial', 20))
        self.title_label.adjustSize()
        self.title_label = QLabel('Upload Your Images Below', self)
        self.title_label.move(100, 75)
        self.title_label.setFont(QFont('Arial', 10))
        self.title_label.adjustSize()
        
        self.button1 = QPushButton("Select Your Folder of Images to Upload Here", self)   
        self.button1.clicked.connect(self.get_image_files)
        self.button1.move(50, 150)
        
# single mode - results
class Page4Single(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(Page4Single, self).__init__(parent)
        
        self.title_label = QLabel('DCiFR', self)
        self.title_label.move(50, 30)
        self.title_label.setFont(QFont('Arial', 20))
        self.title_label.adjustSize()

        self.title_label = QLabel('Results', self)
        self.title_label.move(100, 75)
        self.title_label.setFont(QFont('Arial', 15))
        self.title_label.adjustSize()
        self.title_label = QLabel('Here are the results for the IMAGE you uploaded:', self)
        self.title_label.move(100, 125)
        self.title_label.setFont(QFont('Arial', 10))
        self.title_label.adjustSize()
        
        results_label = QLabel("Please check your working directory for a\nCSV results file", self)
        results_label.move(125, 150)
        results_label.setFont(QFont('Arial', 15))
        results_label.adjustSize()

# batch mode - results
class Page4Batch(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(Page4Batch, self).__init__(parent)
        
        self.title_label = QLabel('DCiFR', self)
        self.title_label.move(50, 30)
        self.title_label.setFont(QFont('Arial', 20))
        self.title_label.adjustSize()

        self.title_label = QLabel('Results', self)
        self.title_label.move(100, 75)
        self.title_label.setFont(QFont('Arial', 15))
        self.title_label.adjustSize()
        self.title_label = QLabel('Here are the results for the IMAGES you uploaded:', self)
        self.title_label.move(100, 125)
        self.title_label.setFont(QFont('Arial', 10))
        self.title_label.adjustSize()
        
        results_label = QLabel("Please check your working directory for a\nCSV results file", self)
        results_label.move(125, 150)
        results_label.setFont(QFont('Arial', 15))
        results_label.adjustSize()
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = Wizard()
    wizard.show()
    app.exec_()  

