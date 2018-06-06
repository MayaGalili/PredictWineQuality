# -*- coding: utf-8 -*-
"""
Machine learning test

Predict wine quality with machine learning models

Created on Wed Jun  6 10:51:30 2018
@author: Maya Galili
"""

''' load lab '''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


''' load data '''
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
all_df = pd.read_csv(dataset_url,delimiter=';')

features_to_use  = ["volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

X_df = all_df[features_to_use]
Y_df = all_df["quality"]

''' check data '''
print(all_df.describe())
X_df.info()
Y_df.hist()
sns.pairplot(all_df)

''' collect models '''
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('AB', AdaBoostClassifier()))
models.append(('LSVM', LinearSVC()))

''' evaluate each model in turn '''
seed = 7
scoring_type = 'accuracy'
X_tr, X_ts, Y_tr, Y_ts = model_selection.train_test_split(X_df, Y_df, test_size=0.20, random_state=seed)
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_tr, Y_tr, cv=kfold, scoring=scoring_type)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

''' plot results '''
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

''' make prediction and test results'''
classifier_type = RandomForestClassifier()
classifier_type.fit(X_tr, Y_tr)
predictions = classifier_type.predict(X_ts)
print('accuracy score: ')
print(accuracy_score(Y_ts, predictions))
print('confusion matrix: ' )
print(confusion_matrix(Y_ts, predictions))
print('classification report: ' )
print(classification_report(Y_ts, predictions))
