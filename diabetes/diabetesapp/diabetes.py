import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os

plt.style.use('seaborn-darkgrid')
class Diabetes:
    def __init__(self):
        self.path=os.getcwd()
        self.path = self.path + "/diabetesapp/data/diabetes.csv"
       
        self.df = pd.read_csv(self.path)
        print(self.df.sample())

    def getDatasetInfo(self):
        return self.df.shape, self.df.columns, self.df.dtypes

    def getDataset(self):
        return self.df

    def dataPreprocessing(self):
        # replacing 0 from 3 columns - BloodPressure, SkinThickness, BMI by NaN
        cols = ['BloodPressure','SkinThickness','BMI']
        self.df[cols] = self.df[cols].replace(0,np.NaN)    

    def splittingDataset(self):
        self.df = self.df.dropna()
        self.train, self.test = train_test_split(self.df, test_size= 0.2, random_state= 42)

    def fillMissingValues(self):
        self.train['BloodPressure'].fillna(self.train['BloodPressure'].mean(), inplace = True)
        self.train['SkinThickness'].fillna(self.train['SkinThickness'].median(), inplace = True)
        self.train['BMI'].fillna(self.train['BMI'].median(), inplace = True)
        self.train.loc[self.train['Insulin'] > 400, 'Insulin'] = self.train['Insulin'].median() 
        self.train.loc[self.train['SkinThickness'] > 50, 'SkinThickness'] = self.train['SkinThickness'].median() 
        self.train.loc[self.train['Pregnancies'] > 13, 'Pregnancies'] = 13
        self.test.loc[self.test['Insulin'] > 400, 'Insulin'] = self.test['Insulin'].median() 
        self.test.loc[self.test['SkinThickness'] > 50, 'SkinThickness'] = self.test['SkinThickness'].median()
    def dataVisualization(self):
        data = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',\
                'BMI','DiabetesPedigreeFunction','Age','Outcome']
        
        plt.figure(figsize = (12,8))
        plt.tight_layout()
        plt.suptitle('Exploratory Analysis on Various Diabetes Features')
        for i,col in enumerate(self.train):
            plt.subplot(3,3,i+1)
            sns.histplot(data = self.train, x=col, kde=True)
            plt.xlabel(col, fontsize = 10)
            plt.xticks(fontsize = 10)   
        plt.savefig('graphs/eda1.png')     
        plt.show()    
        self.fillMissingValues()
        # plt.figure(figsize = (12,8))
        # plt.tight_layout()
        # plt.suptitle('Exploratory Analysis on Various Diabetes Features')
        # for i,col in enumerate(self.train):
        #     plt.subplot(3,3,i+1)
        #     sns.histplot(data = self.train, x=col, kde=True)
        #     plt.xlabel(col, fontsize = 10)
        #     plt.xticks(fontsize = 10)    
        # plt.show()    
        plt.figure(figsize=(12,8))
        plt.tight_layout()
        plt.suptitle('Boxplots on Various Diabetes Features')
        for i,col in enumerate(self.train):
            plt.subplot(3,3,i+1)
            sns.boxplot(data = self.train,x=col)
            plt.xlabel(col, fontsize=10)
            plt.xticks(fontsize=10)
        plt.savefig('graphs/eda2.png')     
        plt.show()
        sns.set(font_scale=1.15)
        plt.figure(figsize=(12, 8))
        ax = plt.axes()
        ax.set_title('Co-relation Matrix between Diabetes Features')
        sns.heatmap(self.train.corr(),vmax=.8, linewidths=0.01, square=True,annot=True,cmap='YlGnBu',linecolor="black")
        plt.savefig('graphs/eda3.png')
        plt.show()
        ax = plt.axes()
        ax.set_title('Distribution of Diabetes outcomes : 0 No Diabetes, 1 Diabetes')
        sns.countplot(x = 'Outcome', data = self.train)
        plt.savefig('graphs/eda4.png')
        plt.show()

    def dataSlicing(self):
        
        self.x_train = self.train.drop('Outcome', axis = 1)
        self.y_train = self.train['Outcome']
        self.x_test = self.test.drop('Outcome', axis = 1)
        self.y_test = self.test['Outcome']

    def modelTraining(self):
        # Data and Model arrays
        self.models = ['Logistic Regression','Random Forest','K-Nearest Neighbour','Support Vector Machine']
        self.accs =[]
        # Logistic regression Accuracy
        log_reg = LogisticRegression(max_iter = 1000)
        log_reg.fit(self.x_train, self.y_train)
        y_pred_l = log_reg.predict(self.x_test)
        logistic_acc = accuracy_score(self.y_test, y_pred_l)
        logistic_acc = logistic_acc * 100
        self.accs.append(round(logistic_acc,2))
        print(f'Logistic regression accuracy = {logistic_acc:.4f}')  
        # Random Forest
        RFC = RandomForestClassifier(n_estimators=180, max_depth = 3,max_features = 'auto', verbose = 1, random_state=42)
        RFC.fit(self.x_train,self.y_train)
        y_pred_r = RFC.predict(self.x_test)
        rfc_acc = accuracy_score(self.y_test, y_pred_r)
        rfc_acc = rfc_acc * 100
        self.accs.append(round(rfc_acc,2))
        print(f'Random forests classifier accuracy = {rfc_acc:.4f}')
        # KNN
        KNN = KNeighborsClassifier(n_neighbors = 90)
        KNN.fit(self.x_train,self.y_train)
        y_pred_k = KNN.predict(self.x_test)
        knn_acc = accuracy_score(self.y_test, y_pred_k) 
        knn_acc = knn_acc * 100
        self.accs.append(round(knn_acc,2))
        print(f'KNN classifier accuracy = {knn_acc:.4f}')
        # SVM
        svm = SVC(C = 1.7, kernel= 'linear')
        svm.fit(self.x_train,self.y_train)
        y_pred_s = svm.predict(self.x_test)
        svc_acc = accuracy_score(self.y_test, y_pred_s)
        svc_acc = svc_acc * 100
        self.accs.append(round(svc_acc))
        print(f'SVM classifier accuracy = {svc_acc:.4f}')
        colors = ['red','green','blue','purple']
        print(self.models)
        print(self.accs)
        plt.title('Analysis of Training Models on Diabetes Predictions Accuracy')
        plt.xlabel('Model')
        plt.ylabel('Accuracy Percentage')
        plt.bar(self.models,self.accs,color=colors)
        #plt.savefig('graphs/compml.png')
        #plt.show()
        return self.accs,self.models

# db = Diabetes()
# shape, columns, dtypes = db.getDatasetInfo()
# df = db.getDataset()    
# print('--- General information of Dataset ---')   
# print('1. No of Rows and columns ::') 
# print(shape)
# print()
# print('2. Coumns name ::')
# print(columns)
# print()
# print('3. Columns Data Types ::')
# print(dtypes)
# print()
# print('--- Sample records of Dataset ---')
# print(df.sample(5))
# print()
# print('--- Missing Values in Dataset ---')
# print(df.isnull().sum())

# db.dataPreprocessing()
# db.splittingDataset()
# # db.dataVisualization()
# db.dataSlicing()
# db.modelTraining()