## we are only using Naive Bayes classifier of Gaussian Type
## other types are 
##					1. Bernoulli
##					2. Multinomial

import numpy as np 
import matplotlib.pyplot as plt 


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification 
from sklearn.datasets import load_breast_cancer 
from sklearn.datasets import make_regression
from sklearn.datasets import make_friedman1
from sklearn.naive_bayes import GaussianNB

## making synthetic datasets  for simple regression


x_r1 , y_r1 = make_regression(n_samples= 100 , n_features = 1 , n_informative = 1 , 
			 bias = 150 ,  noise = 42  , random_state = 0)

plt.scatter(x_r1 , y_r1 , marker = "o" , s = 50)
plt.title("Sample regression problem for one input variable")
#plt.show()

## making synthetics datasets for complex regression



x_f1 , y_f1 = make_friedman1(n_samples = 150 , n_features = 7 ,
							random_state = 0)
#print (x_f1)
#print (x_f1[:,1])
plt.title("complex regression problem with one input variable")
plt.scatter(x_f1[: , 2] , y_f1 , marker = 'o' , s = 50 )
#plt.show()

## synthetic data for classification (binary) with 2 features

x_c1 , y_c1 = make_classification(n_samples = 100 , n_features  = 2 , n_redundant = 0 ,
				n_informative =  2 , n_clusters_per_class = 1 , flip_y = 0.1 ,
				class_sep = .5 , random_state = 0 )
plt.title("binary classification with 2 features")
plt.scatter(x_c1[: , 0] , x_c1[: , 1] , marker = "o" , c = y_c1 , s = 50 )
#plt.show()

x_train , x_test , y_train , y_test = train_test_split(x_c1 , y_c1 , random_state = 0)

nb_clf =  GaussianNB()
nb_clf.fit(x_train , y_train )
## for bianry classification
print ("scores for binary classification ")
print ("test scores")
print (nb_clf.score(x_test , y_test))
print ("train scores")
print (nb_clf.score(x_train , y_train))
#print (load_breast_cancer())
cancer_data = load_breast_cancer()
x_data , y_data =  cancer_data.data , cancer_data.target

x_train , x_test , y_train , y_test  = train_test_split(x_data , y_data)
nb_clf.fit(x_train , y_train)
print ("scores for train data")
print (nb_clf.score(x_train , y_train))
print ("scores for test data")
print (nb_clf.score(x_test , y_test))
