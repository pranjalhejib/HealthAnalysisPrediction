import cv2
import numpy as np
import time
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import tkinter as tk 
from tkinter import filedialog
from keras.models import load_model


def testingProcess(self):
        model = load_model('./NN.h5')
        classes = ['Brain Tumor', 'Healthy']
        while 1:
                bt_result = 'Healthy Brain'
                bt_stage = ' '
                bt_symptoms = ' '

                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename()
                img = cv2.imread(file_path)
                height, width = img.shape[:2]
                img = cv2.resize(img, (100,100))
                # predict!
                roi_X = np.expand_dims(img, axis=0)
                predictions = model.predict(roi_X)
                #print(np.argmax(predictions[0]))
                result_index = np.argmax(predictions[0])
        ##        print (classes[result_index]) 
        ##        print(acc)
                if(result_index == 0):
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                        count = cv2.countNonZero(blackAndWhiteImage)
                        per =(count * 100)/ (height * width)
                        bt_result = 'Brain Tumor Detected'
                        if(per<1):
                                bt_stage = 'Brain Tumor of Stage 1'
                                bt_symptoms = 'The tumor grows slowly and rarely spreads into nearby tissues. It may be possible to completely remove the tumor with surgery.'

                        elif(per<3):
                                bt_stage = 'Brain Tumor of Stage 2'
                                bt_symptoms = 'The tumor grows slowly but may spread into nearby tissues or recur.'
                        else:
                                bt_stage = 'Brain Tumor of Stage 3'
                                bt_symptoms = 'The tumor grows quickly, is likely to spread into nearby tissues, and the tumor cells look very different from normal cells.'

                print(bt_result)
                print(bt_stage)
                print(bt_symptoms)
                print('____________________________')


