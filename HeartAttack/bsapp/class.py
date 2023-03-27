from pandas import read_csv, DataFrame
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os

class Classification:

	def __init__(self):

		pd.set_option('display.max_columns', None)
		#np.set_printoptions(threshold=np.nan)
		np.set_printoptions(precision=3)
		sns.set(style="darkgrid")
		plt.rcParams['axes.labelsize'] = 14
		plt.rcParams['xtick.labelsize'] = 7
		plt.rcParams['ytick.labelsize'] = 12
		self.names = ['age', 'sex',\
		 'cp','trestbps','chol','fbs','restecg','thalach','exang',\
		     'oldpeak','slope','ca','thal','target']
		
		self.path=os.getcwd()
		# print(path)     
		#self.ds = pd.read_csv("data/heart.csv")
		self.ds = pd.read_csv("f:/heart.csv")
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
		ds1=DataFrame([self.ds.age,self.ds.sex,self.ds.cp,self.ds.target]) 
		#,ds.chol,ds.thalach,target])
		scatter_matrix(ds1, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
		pyplot.show()

		
	def classificationBoxPlot(self):		
		pyplot.boxplot(self.results, labels=self.names)
		pyplot.title('Algorithm Comparison')
		pyplot.show()

	def edaTarget(self):
		# y = self.ds["target"]
		# print(y)
		# sns.countplot(y)
		# # target_temp = self.ds.target.value_counts()
		
		# pyplot.title("Data Exploration on Variable Target")
		# pyplot.savefig(self.path+"/static/graphs/dataexp_target.png")
		
		# pyplot.show()
		age = self.ds["age"]
		sns.countplot(age)
		
		pyplot.title("Data Exploration on Variable Age")
		pyplot.savefig(self.path+"/static/graphs/dataexp_age.png")
		pyplot.show()

		# sex = self.ds["sex"]
		# # sns.palplot(sns.color_palette("Paired", 2))
		# sns.countplot(sex)
		
		# pyplot.title("Data Exploration on Variable Sex")
		# pyplot.savefig(self.path+"/static/graphs/dataexp_sex.png")
		# pyplot.show()

		# cp = self.ds["cp"]
		# # sns.palplot(sns.color_palette("Paired", 2))
		# sns.countplot(cp)
		
		# pyplot.title("Data Exploration on Chest Pain")
		# pyplot.savefig(self.path+"/static/graphs/dataexp_cp.png")
		# pyplot.show()

		# ch = self.ds["chol"]
		# # sns.palplot(sns.color_palette("Paired", 2))
		# pyplot.boxplot(ch)
		# pyplot.xlabel("Groups of Patients")
		# pyplot.ylabel("Serum cholesterol in mg/dl")
		
		# pyplot.title("Data Exploration using Histogram on Cholesterol Values")
		# pyplot.savefig(self.path+"/static/graphs/dataexp_cholbox.png")
		# pyplot.show()

		# th = self.ds["thalach"]
		# # sns.palplot(sns.color_palette("Paired", 2))
		# pyplot.boxplot(th)
		# pyplot.xlabel("Groups of Patients")
		# pyplot.ylabel("Hdeart Rate")
		
		# pyplot.title("Data Exploration using Histogram on Hdeart Rate Values")
		# pyplot.savefig(self.path+"/static/graphs/dataexp_talachbox.png")
		# pyplot.show()

		# th = self.ds["thalach"]
		# # sns.palplot(sns.color_palette("Paired", 2))
		# dt = DataFrame([ch,th])
		# box = plt.boxplot(dt, patch_artist=True)
 
		# colors = ['purple', 'tan']
 
		# for patch, color in zip(box['boxes'], colors):
		# 	patch.set_facecolor(color)
    		
			
		# pyplot.title("Data Exploration using Histogram on Max Heart Rate")
		# pyplot.savefig(self.path+"/static/graphs/dataexp_thalachbox.png")
		# pyplot.show()
	def distPlot(self):
		sns.distplot(self.ds["thal"])
		pyplot.title('Distribution Plot on Heart Rate')
		pyplot.savefig(self.path+"/static/graphs/explore_thal.png")
		pyplot.show()
		sns.distplot(self.ds["chol"],color='green')
		pyplot.title('Distribution Plot on Cholesterol Levels')
		pyplot.savefig(self.path+"/static/graphs/explore_chol.png")
		pyplot.show()	

	def classificationModels(self):
		from sklearn.model_selection import train_test_split
		from sklearn.metrics import accuracy_score
		from sklearn.linear_model import LogisticRegression
		import time

		# split dataset into training and testing sets
		predictors = self.ds.drop("target",axis=1)
		target = self.ds["target"]

		X_train,X_test,Y_train,Y_test = \
		     train_test_split(predictors,target,test_size=0.20,random_state=0)

		results=[]
		times=[]
		methods=['Logistic Regression','Naive Bayes Classifier','Support Vector Machine',\
				'K Nearest Neighbourhood','DecisionTreeClassifier',\
				'Random Forest Classifier']		
		st=time.time()
		lr = LogisticRegression()
		lr.fit(X_train,Y_train)
		Y_pred_lr = lr.predict(X_test)
		score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
		print(score_lr)
		results.append(score_lr)
		et=time.time()
		tt=(et-st) * 4
		times.append(tt)

		st=time.time()
		nb = GaussianNB()
		nb.fit(X_train,Y_train)
		Y_pred_nb = nb.predict(X_test)
		score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
		print(score_nb)
		results.append(score_nb)
		et=time.time()
		tt=(et-st) * 4
		times.append(tt)

		st=time.time()		
		from sklearn import svm
		sv = svm.SVC(kernel='linear')
		sv.fit(X_train, Y_train)
		Y_pred_svm = sv.predict(X_test)
		score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
		print(score_svm)
		results.append(score_svm)
		et=time.time()
		tt=(et-st) * 4
		times.append(tt)

		st=time.time()
		knn = KNeighborsClassifier(n_neighbors=7)
		knn.fit(X_train,Y_train)
		Y_pred_knn=knn.predict(X_test)
		score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
		print(score_knn)
		results.append(score_knn)
		print(results)
		et=time.time()
		tt=(et-st) * 4
		times.append(tt)

		st=time.time()
		dt = DecisionTreeClassifier()
		dt.fit(X_train,Y_train)
		Y_pred_dt=dt.predict(X_test)
		score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
		print(score_dt)
		results.append(score_dt)
		print(results)
		et=time.time()
		tt=(et-st) * 4
		times.append(tt)

		st=time.time()
		from sklearn.ensemble import RandomForestClassifier
		rf = RandomForestClassifier()
		rf.fit(X_train,Y_train)
		Y_pred_rf=rf.predict(X_test)
		score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
		print(score_rf)
		results.append(score_rf)
		print(results)
		et=time.time()
		tt=(et-st) * 4
		times.append(tt)

		

		methods1=['LR','NB','SVM','KNN','DT','RF']
		# colors=['lightblue','orange','lightgreen','violet','lightgray','pink']
		# variance = [1, 2, 7, 4, 2, 3]
		# x_pos = [i for i, _ in enumerate(methods1)]				
		# pyplot.xticks(x_pos, methods1)
		# pyplot.bar(x_pos,results,color=colors,yerr=variance)
		# pyplot.xlabel("Classification Method")
		# pyplot.ylabel("Accuracy Percentage")
		# pyplot.title("Comparitive Analysis of Classification Methods")
		
		# pyplot.savefig(self.path+"/static/graphs/compare.png")
		# pyplot.show()

		ind = np.arange(1,7) 
		width = 0.35 
		# plt.bar(ind, results, width, label='Accuracy')
		# plt.bar(ind + width	, times, width, label='Exec Time')
		plt.bar(methods1,times,label="Excecution Time")
		plt.title('Accuracy vs Excecution Time')
		plt.legend(loc='best')
		plt.show()
		return results, methods

obj = Classification()
#obj.edaTarget()
obj.classificationModels()





# array = self.ds.values
		
# 		y = array[:,4]
# 		X_train, X_test, Y_train, Y_test = \
# 			train_test_split(X, y, test_size=0.20, random_state=1)
		
# 		models = []
# 		models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# 		models.append(('LDA', LinearDiscriminantAnalysis()))
# 		models.append(('KNN', KNeighborsClassifier()))
# 		models.append(('CART', DecisionTreeClassifier()))
# 		models.append(('NB', GaussianNB()))
# 		models.append(('SVM', SVC(gamma='auto')))

# 		# evaluate each model in turn
# 		self.results = []
# 		self.names = []

# 		for name, model in models:
# 			kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# 			cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
# 			self.results.append(cv_results)
# 			self.names.append(name)
# 			print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))