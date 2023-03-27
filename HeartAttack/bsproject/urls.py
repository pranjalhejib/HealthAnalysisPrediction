"""bsproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from django.conf.urls import url
from bsapp.views import index, datasetDetails, dashboard, datasamples
from bsapp.views import statsanalysis, distanalysis,distanalysis1, dataexplore
from bsapp.views import classification, comparison,getpredictions, pred, sample

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^index/',index,name='index'),
    url(r'^datasetDetails/',datasetDetails,name='datasetDetails'),
    url(r'^datasamples/',datasamples,name='datasamples'),
    url(r'^statsanalysis/',statsanalysis,name='statsanalysis'),
    url(r'^distanalysis/',distanalysis,name='distanalysis'),
    url(r'^distanalysis1/',distanalysis1,name='distanalysis1'),
    url(r'^dataexplore/',dataexplore,name='dataexplore'),
    url(r'^classification/',classification,name='classification'),
    url(r'^comparison/',comparison,name='comparison'),
    url(r'^getpredictions/',getpredictions,name='getpredictions'),
    url(r'^$',dashboard,name='dashboard'),
    url(r'^pres/',pred,name='pred'),
    path('sample/',sample,name='sample'),
]
