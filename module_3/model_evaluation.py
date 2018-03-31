import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_digits

## bincount in numpy 
## it is used to count 
x = np.array([1,2,4,5,7,1,5,2,6,1,5,5,6])
## prints the occurances of each number in the numpy array x
print (np.bincount(x))

## load datasets 

dataset = load_digits()
#print (dataset)
x , y = dataset.data , dataset.target

print (np.bincount(dataset['target']))
print (dataset['target_names'])

## combinning the two target and target_names

dict_target = dict(zip(dataset['target_names'] , np.bincount(dataset['target'])))
print (dict_target)

## this shows that we nealy equal amount of data for each number 
## this is not imbalanced data

## for imbalanced data

## any change in  the original data will be effected so

y_imbalance = dataset['target'].copy() 
print (y_imbalance)

## this makes our data imbalanced
y_imbalance[y_imbalance != 1] = 0
#print (y_imbalance)
print (np.bincount(y_imbalance))

x_train , x_test , y_train , y_test = train_test_split(x , y_imbalance , test_size = .25 , random_state = 0)

from sklearn.svm import SVC
clf_svm = SVC( kernel = 'rbf' , C = 1)
clf_svm.fit(x_train , y_train )
print (clf_svm.score(x_test , y_test ))

## making a dummy classifier for this
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier( strategy = 'most_frequent').fit(x_train , y_train)
print (dummy.score(x_test , y_test))


## DummyClassifier and our SVM model with kernel = rbf have almost same 
## accuracy results so we need a different model

clf_svm = SVC( kernel = "linear" , C = 1)
clf_svm.fit(x_train , y_train)
print (clf_svm.score(x_test , y_test))
