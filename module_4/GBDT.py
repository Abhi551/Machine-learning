## Another method in ensemble is GradientBoostingClassifier
## which assumes every DT to be a weak learner and makes improvement on every weak learner

import numpy as  np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import make_blobs , load_breast_cancer

## on artificial datasets 
x_d ,  y_d =  make_blobs(n_samples =  100 , n_features = 2 , centers = 8 ,
						cluster_std = 1.3 , random_state = 4)
y_d = y_d%2
plt.figure()
plt.title("sample binary classification problem with non-linearly separable classes")
plt.scatter(x_d[:,0], x_d[:,1], c=y_d, marker= 'o', s=50)
plt.show()

x_train , x_test , y_train , y_test =  train_test_split(x_d , y_d , random_state = 0)
fig , subaxes  = plt.subplots(1,1 , figsize = (6,6))

## GradientBoostingClassifier
clf =  GradientBoostingClassifier()

## fitting the training set 
clf.fit(x_train , y_train)

print ("test score on GBDT is = ", clf.score(x_test , y_test))
print ("train score on GBDT is = ", clf.score(x_train , y_train))
## plotting the GBDT  
title = 'GBDT, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf ,  x_train , y_train , x_test ,  y_test , 
											title , subaxes )
plt.show()

## GBDT on fruits_datasets

fruits_df =  pd.read_csv("fruit_data_with_colors.txt" , delimiter = "\t")

## selecting the features from the datasets
x_data = fruits_df[['mass' , 'width' , 'color_score' , 'height']]
y_data = fruits_df['fruit_label']

## differecnce in normal data and as_matrix()
print x_data.head()
print x_data.head().as_matrix()

## Default Settings
## using normal data
x_train , x_test , y_train , y_test = train_test_split( x_data , y_data , random_state = 0)

#print (x_train.head())
clf =  GradientBoostingClassifier()
clf.fit( x_train , y_train)
print ("Scoring rate for training data")
print (clf.score(x_train , y_train))
print ("Scoring rate for test data")
print clf.score(x_test , y_test)

## using as_matrix()
x_train , x_test , y_train , y_test =  train_test_split(x_data.as_matrix() ,  y_data.as_matrix() , random_state = 0)
clf.fit(x_train , y_train)
print ("Scoring rate for training data")
print (clf.score(x_train , y_train))
print ("Scoring rate for test data")
print clf.score(x_test , y_test)

## plotting 

fig , subaxes = plt.subplots( 6 , figsize = (6,32))
title = 'GBDT, complex binary dataset, default settings'
pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
feature_names_fruits = ['mass' , 'width' , 'color_score' , 'height']
for pair, axis in zip(pair_list, subaxes):
    x = x_train[:, pair]
    y = y_train
    
    clf = GradientBoostingClassifier().fit(x, y)
    plot_class_regions_for_classifier_subplot(clf, x , y, None,
                                             None, title, axis,
                                             target_names_fruits)
    
    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])
    
plt.tight_layout()
plt.show()

## real world dataset of load breast cancer 

breast_cancer =  load_breast_cancer()

x_data =  breast_cancer.data 
y_data = breast_cancer.target 

## default settings for GBDT

x_train ,  x_test , y_train , y_test =  train_test_split(x_data , y_data , random_state = 0)
clf = GradientBoostingClassifier(max_depth = 3 , learning_rate = .1)
print ("the default value settings ")
clf.fit(x_train , y_train)
print ("the score on training data is ")
print (clf.score(x_train , y_train))
print ("the score on testing data is ")
print (clf.score(x_test , y_test))

## using the differen values for learning_rate 
learning_rate =  np.linspace(.01 , .1 , 10)
#print (learning_rate)
for rates in learning_rate:
	clf = GradientBoostingClassifier(max_depth = 2 ,  learning_rate = rates)
	clf.fit(x_train , y_train)
	print ("the value of learning_rate " , rates)
	print (clf.score(x_test , y_test))
