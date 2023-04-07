# Main web app for drowsiness detection project

import numpy as np
from flask import Flask, render_template, request, make_response
#from lungs_testing import get
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import load_model

app = Flask(__name__)
model = load_model('NN.h5')

def get(file_path):
        result = 'Healthy Lungs'

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
        per = 0
        if(result_index == 1):
                result = 'Lungs Cancer Detected'

                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                count = cv2.countNonZero(blackAndWhiteImage)
                per =((count * 100)/ (height * width))
                per =  "%.2f" % round(per,2)
                plt.imshow(gray)
        return result,per
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/datasetview')
def datasetview():
    return render_template('datasetview.html')    
   
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/deeplearning')
def deeplearning():
    return render_template('deeplearning.html')

@app.route('/testing')
def testing():
    return render_template('testing.html') 

@app.route('/predictData',methods=['POST','GET'])
def predictData():
    if request.method=='POST':
        type = request.form['type']
        fn = request.files['tfile']

        file = fn.filename
        path="../static/images/"+type+"/"+file
        path1="static/images/"+type+"/"+file
        result,per = get(path1)
    return render_template('predictdata.html', file=file,type=type,path=path,result=result,per=per)        

@app.route('/about')
def about():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)    