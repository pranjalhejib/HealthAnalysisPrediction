# Main web app for drowsiness detection project
import time
from asyncio.windows_events import NULL
from functools import cache
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import matplotlib.pyplot as plt
import os
from flask_caching import Cache
from flask import Flask, render_template, request, make_response
import cv2

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('about.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/datasetview')
def datasetview():
    return render_template('datasetview.html')    

@app.route('/deeplearning')
def deeplearning():
    return render_template('deeplearning.html')

@app.route('/analysis')
def analysis():
    return render_template('visualization.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)    