from django.shortcuts import render
from django.http import HttpResponse
import os
import matplotlib as plt
import cv2
import numpy as np
from keras.models import load_model
import os
import LungApp.lungs_testing as lt
# Create your views here.
# print(os.getcwd())
# model = load_model('static/models/NN.h5')
# print(os.getcwd())
def homepage(request):
    return render(request, 'homepage.html',{})

def index(request):
    return render(request,'index.html',{})

def datasetview(request):
    return render (request,'datasetview.html',{})

def analysis(request):
    return render (request,'analysis.html',{})

def visualization(request):
    return render (request,'visualization.html',{}) 

def deeplearning(request):
    return render (request,'deeplearning.html',{}) 

def testing(request):
    return render (request,'testing.html',{})    

def predictData(request):
    if request.method == 'POST':
        f = request.FILES['tfile']
        cwd = os.getcwd()+'/LungApp/static/uploads/'
        with open(cwd+f.name, 'wb+') as file:  
            for chunk in f.chunks():  
                file.write(chunk) 
     
    imgpath = cwd+f.name           
    result,per = lt.get(imgpath)    
    return render(request,'result.html',{'result':result,'imgpath':f.name})            
        

