"""diabetes URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from diabetesapp.views import home, dashboard, datasamples, statsanalysis, distanalysis 
from diabetesapp.views import distanalysis1, comat, comparison, classification
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',dashboard,name='dashboard'),
    path('/home',home,name='home'),
    path('/datasamples',datasamples,name='datasamples'),
    path('/statsanalysis',statsanalysis,name='statsanalysis'),
    path('/distanalysis',distanalysis,name='distanalysis'),
    path('/distanalysis1',distanalysis1,name='distanalysis1'),
    path('/comat',comat,name='comat'),
    path('/comparison',comparison,name='comparison'),
    path('/classification',classification,name='classification'),
]
