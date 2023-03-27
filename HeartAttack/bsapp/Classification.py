from pandas import read_csv, DataFrame
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os

predictions = None

class Classification:

	def __init__(self):
		self.names = ['age', 'sex',\
		 'cp','trestbps','chol','fbs','restecg','thalach','exang',\
		     'oldpeak','slope','ca','thal','target']
		
		self.path=os.getcwd()
		    
		#self.path=self.path+"/bsapp/data/heart.csv"
		self.path = self.path + "/data/heart.csv"
		print(self.path) 
		self.ds = pd.read_csv(self.path)
		#self.path=os.getcwd()+"/bsapp/data/heart.csv"
		self.path = os.getcwd() + "/data/heart.csv"
		self.dsl = pd.read_csv(self.path)
		# self.ds = pd.read_csv("f://heart.csv")
		self.dsname = 'heart.csv'
		
		self.dtypes = ['Numeric','Categorical','Categorical','Numeric','Numeric',\
		 			   'Categorical','Categorical','Numeric','Categorical','Numeric',\
		 			   'Categorical','Categorical','Categorical','Categorical']
	def getHeading(self):
	 	return self.names
			
	def datasetDetails(self):
		return self.ds, self.dsname ,self.names, self.dtypes

	def statDetails(self):
		print(self.ds.describe())

	def getDataset(self):
		return self.ds

	def getDatasetLess(self):
		return self.dsl, self.names	
	
	def datasetOverview(self):
		return self.ds, self.dsname, self.names, self.dtypes

	def dataVizBoxplot(self):
		self.ds.plot(kind='box', subplots=True, \
			layout=(2,2), sharex=False, sharey=False)
		pyplot.savefig('boxplot.png')
		pyplot.show()	

	def dataVizHist(self):
		self.ds.hist()
		pyplot.savefig('boxplot.png')
		pyplot.show()

	def dataScatterMatrix(self):
		scatter_matrix(self.ds)
		pyplot.show()

	# classification methods

	def classificationModels(self):
		from sklearn.model_selection import train_test_split
		from sklearn.metrics import accuracy_score
		from sklearn.linear_model import LogisticRegression

		# split dataset into training and testing sets
		predictors = self.ds.drop("target",axis=1)
		target = self.ds["target"]

		X_train,X_test,Y_train,Y_test = \
		     train_test_split(predictors,target,test_size=0.20,random_state=0)

		results=[]
		methods=['Logistic Regression','Naive Bayes Classifier','Support Vector Machine',\
				'K Nearest Neighbourhood','DecisionTreeClassifier',\
				'Random Forest Classifier']		
		
		lr = LogisticRegression()
		lr.fit(X_train,Y_train)
		Y_pred_lr = lr.predict(X_test)
		score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
		print(score_lr)
		results.append(score_lr)
		
		nb = GaussianNB()
		nb.fit(X_train,Y_train)
		Y_pred_nb = nb.predict(X_test)
		score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
		print(score_nb)
		results.append(score_nb)
		
		from sklearn import svm
		sv = svm.SVC(kernel='linear')
		sv.fit(X_train, Y_train)
		Y_pred_svm = sv.predict(X_test)
		score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
		print(score_svm)
		results.append(score_svm)
		
		knn = KNeighborsClassifier(n_neighbors=7)
		knn.fit(X_train,Y_train)
		Y_pred_knn=knn.predict(X_test)
		score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
		print(score_knn)
		results.append(score_knn)
		print(results)

		dt = DecisionTreeClassifier()
		dt.fit(X_train,Y_train)
		Y_pred_dt=dt.predict(X_test)
		score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
		print(score_dt)
		results.append(score_dt)
		print(results)

		rf = RandomForestClassifier()
		rf.fit(X_train,Y_train)
		Y_pred_rf=rf.predict(X_test)
		score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
		print(score_rf)
		results.append(score_rf)
		print(results)
		return results, methods

	def getPredictions(self):
		from sklearn.model_selection import train_test_split
		from sklearn.metrics import accuracy_score
		from sklearn.linear_model import LogisticRegression

		# split dataset into training and testing sets
		predictors = self.ds.drop("target", axis=1)
		target = self.ds["target"]

		X_train, X_test, Y_train, Y_test = \
			train_test_split(predictors, target, test_size=0.20, random_state=0)

		results = []
		methods = ['Logistic Regression', 'Naive Bayes Classifier', 'Support Vector Machine', \
				   'K Nearest Neighbourhood', 'DecisionTreeClassifier', \
				   'Random Forest Classifier']

		lr = LogisticRegression()
		lr.fit(X_train, Y_train)
		Y_pred_lr = lr.predict(X_test)
		score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
		print(score_lr)
		results.append(score_lr)

		nb = GaussianNB()
		nb.fit(X_train, Y_train)
		Y_pred_nb = nb.predict(X_test)
		score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)
		print(score_nb)
		results.append(score_nb)

		from sklearn import svm
		sv = svm.SVC(kernel='linear')
		sv.fit(X_train, Y_train)
		Y_pred_svm = sv.predict(X_test)
		score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)
		print(score_svm)
		results.append(score_svm)

		knn = KNeighborsClassifier(n_neighbors=7)
		knn.fit(X_train, Y_train)
		Y_pred_knn = knn.predict(X_test)
		score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)
		print(score_knn)
		results.append(score_knn)
		print(results)

		dt = DecisionTreeClassifier()
		dt.fit(X_train, Y_train)
		Y_pred_dt = dt.predict(X_test)
		score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
		print(score_dt)
		results.append(score_dt)
		print(results)

		rf = RandomForestClassifier()
		rf.fit(X_train, Y_train)
		Y_pred_rf = rf.predict(X_test)
		score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
		print(score_rf)
		results.append(score_rf)
		print(results)
		print(Y_pred_rf)
		print(Y_test.head(5))
		print(Y_test.shape)
		return results, methods, Y_pred_rf, Y_test,self.names

	def getPredictionValue(self,list):
		import numpy as np
		arr = np.array(list,dtype=np.float32).reshape(1,-1)
		print(arr)

		predictors = self.ds.drop("target", axis=1)
		target = self.ds["target"]

		X_train, X_test, Y_train, Y_test = \
			train_test_split(predictors, target, test_size=0.20, random_state=0)
		print(X_test)
		rf = RandomForestClassifier()
		rf.fit(X_train, Y_train)
		Y_pred_rf = rf.predict(X_test)
		
		temp = X_test[1:2]
		print(temp)
		Y_pred_rf = rf.predict(temp)
		score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
		print(score_rf)
		return Y_pred_rf, score_rf

	def classificationBoxPlot(self):		
		pyplot.boxplot(self.results, labels=self.names)
		pyplot.title('Algorithm Comparison')
		pyplot.show()

list = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
obj = Classification()
obj.getPredictionValue(list)