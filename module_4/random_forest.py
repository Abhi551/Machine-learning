## Random Forests are ensembled Decision Trees
## ensembled methods are used for
import numpy as np 
import matplotlib.pyplot as plt 

## for artificial datasets of make_blobs

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import load_breast_cancer , make_blobs

x_d ,  y_d  =  make_blobs(n_samples = 100 , n_features =  2 , centers = 8 , cluster_std = 1.3 
						,random_state = 4)
y_d = y_d%2

x_train , x_test , y_train , y_test =  train_test_split(x_d , y_d , random_state = 0 )

clf = RandomForestClassifier()

clf.fit(x_train , y_train)
print (clf.score(x_train , y_train))
print (clf.score(x_test , y_test))

fig , subaxes = plt.subplots(1,1 , figsize = (6,6) )
title = "Random  Forests"
#plot_class_regions_for_classifier_subplot(clf ,  x_train , y_train ,  x_test , y_test , title  , subaxes)
#plt.show()



## for fruits_datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import pandas as pd

## fruits datasets
fruits_df =  pd.read_csv("fruit_data_with_colors.txt" , delimiter = "\t")
print (fruits_df.head())

x_fruits = fruits_df[["width" , "color_score" , "mass" , "height"]]
y_fruits = fruits_df["fruit_label"]

x_train , x_test , y_train , y_test = train_test_split(x_fruits , y_fruits , random_state = 0)

clf = RandomForestClassifier()

clf.fit(x_train , y_train)
print (clf.score(x_train , y_train))
print (clf.score(x_test , y_test ))

## on real world data of load_breast_cancer

data = load_breast_cancer()

#print (data)

x , y = data.data , data.target
x_train , x_test , y_train , y_test =  train_test_split(x,y)

clf =  RandomForestClassifier()

clf.fit(x_train , y_train)

print (clf.score(x_train , y_train))
print (clf.score(x_test , y_test))

score_list = []

## changing the default settings of Random Forest
for value in [ 1 , 2 , 5 , 8 ]:
	for trees in [5 , 8 , 10 , 11 , 12] :
		for depth in [ 2 , 3  , 4]:
			f_list= []
			clf =  RandomForestClassifier(max_features = value ,  n_estimators = trees , max_depth = depth , random_state = 0)
			clf.fit(x_train , y_train)
			f_list.append(trees )
			f_list.append(depth )
			f_list.append(value )
			f_list.append(clf.score(x_train , y_train))
			f_list.append(clf.score(x_test , y_test))
			score_list.append(f_list)

## getting the list of settings for which the best result will be printed
score_list = np.array(score_list)
#print (score_list.shape)
print (" maximum score is ")
print (max(score_list[:,4]))
for score in score_list :
	if score[4]==max(score_list[:,4]):
		print (score)


	