from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from diabetesapp.diabetes import Diabetes
import os

def home(request):
    path = os.getcwd()
    print(path)
    
    obj = Diabetes()
    dsname='Diabetes'
    shape, columns, dtypes = obj.getDatasetInfo()
    ds = obj.getDataset()
    nr = ds.shape[0]
    nc = ds.shape[1]
    return render(request,'home.html',\
		         {'dsname':dsname,'nc':nc, 'nr':nr,\
		         'names':columns,'dt':dtypes,'nums':range(1,10)})

def dashboard(request):
    return render(request,'dashboard.html',{})

def datasamples(request):
	obj = Diabetes()
	ds = obj.getDataset()
	shape, columns, dtypes = obj.getDatasetInfo();
	return render(request,'datasamples.html',\
		         {'names':columns,'ds':ds.head(20)})

def statsanalysis(request):
	obj = Diabetes()
	
	ds = obj.getDataset()
	return render(request,'statsanalysis.html',{'ds':ds})

def distanalysis(request):
	return render(request,'distanalysis.html')

def distanalysis1(request):
	return render(request,'distanalysis1.html')

def comat(request):
	return render(request,'comat.html')

def comparison(request):
	return render(request,'comparison.html')

def classification(request):
	obj = Diabetes()
	obj.splittingDataset()
	obj.fillMissingValues()
	obj.dataSlicing()
	res, met = obj.modelTraining()
	return render(request,'classification.html',{'res':res,'met':met,'nums':range(1,5)})

def healthapp(request):
	return render(request,'healthapp.html',{})