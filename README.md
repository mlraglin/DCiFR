# DCiFR

![](https://github.com/peter1125/DCiFR/blob/main/logo.png)

DCiFR (Demographic Characteristics in Facial Recognition) is a wrapper software that allows you to run deep learning models to parse demographic characteristics from an image. This open-source wrapper software written in Python has a GUI that will allow you to run complex models without any knowledge of coding. This includes functions from [deepface](https://github.com/serengil/deepface) and [fairface](https://github.com/dchen236/FairFace) and is built with [PyQT5](https://pypi.org/project/PyQt5/) to provide the GUI. This wrapper is hosted on Docker to allow for easy use on any operating system.

## Getting Started

Dependencies for running DCiFR include deepface, and PyQT5.

Run
```
pip install deepface & pyqt5
```
to get started. 


To fire up the GUI, in your terminal type:
```
python3 dcifr_code.py
```

Jupyter Notebook version is also available in this repo - *dcifr_code.ipynb*.

## DeepFace Attributes


Based on faces within images, DCIFR's DeepFace pipeline reports results of four attributes: age, emotion, gender, and race. 

+ Age - Predicted age will fall between 0 - 100. 
+ Emotion - One of seven possible emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
+ Gender - Reports either man or woman.
+ Race - The software predicts the probability of falling into one of seven race categories: Asian, black, Indian, Latino/Hispanic, Middle Eastern, or white. The results show the racial category with the highest probability.

More information on the attributes and how they are modeled can be found [here](https://pypi.org/project/deepface/). 


## FairFace Attributes


Based on faces within images, DCiFR's FairFace pipeline reports results of eight attributes: race, race4, gender, age, race_scores_fair, race_scores_fair_4, gender_scores_fair, and age_scores_fair.

+ Race - Predicted probability of falling into one of seven race categories:  White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, or Middle Eastern.
+ Race4 - Predicted probability of falling into one of four race categories: White, Black, Asian, or Indian.
+ Gender - Reports either male or female.
+ Age - Predicted age will fall within the following ranges: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, or 70+.
+ Race_scores_fair - The model confidence score for predicting race.
+ Race_scores_fair_4 - The model confidence score for predicting race4.
+ Gender_scores_fair - The model confidence score for predicting gender.
+ Age_scores_fair - The model confidence score for predicting age.

More information on the attributes and how they are modeled can be found [here](https://github.com/dchen236/FairFace). 


## Output
The results will be saved in a DCIFR folder within the user's Documents as *dcifr_Deepface_results* or *dcifr_Fairface_results* with the date and time of creation attached to the end of the file name.

## Reference

+ [deepface](https://github.com/serengil/deepface)
+ [fairface](https://github.com/dchen236/FairFace)
+ [PyQT5](https://pypi.org/project/PyQt5/)

## License

    DCiFR: Demographic Characteristics in Facial Recognition.
    Copyright (C) 2021  Melina Raglin, Eungang (Peter) Choi, Erick Axxe

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
