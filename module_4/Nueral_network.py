from sklearn.neural_network import MLPClassifier , MLPRegressor
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs ,  load_breast_cancer , make_regression
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns

## different functions used in Nueral Network
x_range = np.linspace(-2 , 2 , 200)

plt.plot(x_range , np.maximum(x_range , 0) , label = "relu")
plt.plot(x_range , np.tanh(x_range) , label = "tanh")
plt.plot(x_range , 1/(1 + np.exp(-x_range) ) , label = "logistic")
plt.plot(x_range ,[0 for i in range(200)] , color = "grey" )
plt.legend()
plt.title("neural_network plots")
plt.show()

## one hidden layer in MLP classifiers

x_d ,  y_d  =  make_blobs(n_samples = 100 , n_features =  2 , centers = 8 , cluster_std = 1.3 
						,random_state = 4)
y_d = y_d%2

fig , subaxes =  plt.subplots( 3 , 1 , figsize=(6,18))

x_train , x_test , y_train , y_test = train_test_split (x_d ,  y_d , random_state = 0)

fig , subaxes = plt.subplots( 3 , 1 , figsize = (6 , 18 ))

for units , axis in zip([1 , 10 , 100] ,  subaxes):
	nueral_clf =  MLPClassifier(hidden_layer_sizes = [units] , solver = "lbfgs" , random_state = 0 )
	nueral_clf.fit(x_train , y_train)
	print ("nueral networks for hidden layer of size %d" %(units))
	print (nueral_clf.score(x_train , y_train))
	print (nueral_clf.score(x_test ,  y_test))

	## plotting the results 
	title =  "plot for hidden layer of size %d " %(units)
	plot_class_regions_for_classifier_subplot(nueral_clf , x_train , y_train , x_test , y_test , title  , axis )
	plt.tight_layout()
	


## using two hidden layers for the MLPClassifier
x_train , x_test , y_train , y_test = train_test_split (x_d , y_d , random_state = 0)

for unit in [10 , 100]:
	nnclf =  MLPClassifier( hidden_layer_sizes = [unit, unit] , solver = "lbfgs" , random_state = 0)
	nnclf.fit(x_train , y_train)

	print ("for 2 hidden layer of sizes %d each"%unit)
	print (nnclf.score(x_train , y_train))
	print (nnclf.score(x_test , y_test))


## using regularization parameter "alpha" in MLPClassifier
## as regularization parameter alpha is incresed accuracy on testing data increases

fig , subaxes = plt.subplots( 4 , 1 , figsize = (6,23))

for this_alpha , axis in zip([.001 , .1 , 1.0 , 5.0 ] , subaxes):
	for units in [ 50 ] :
		nnclf =  MLPClassifier (solver = "lbfgs" ,  activation = "tanh" , alpha = this_alpha ,
							hidden_layer_sizes = [units] , random_state = 0)
		nnclf.fit(x_train , y_train)
		print ("neural_network classifier score for hidden_layer_sizes = %d and aplha = %f  " % (units , this_alpha))
		print (nnclf.score(x_train , y_train))
		print (nnclf.score(x_test , y_test))

## using different activation function on same dataset

for activation_func in ["relu" ,  "tanh" , "logistic"]:
	nnclf =  MLPClassifier(solver = "lbfgs" , hidden_layer_sizes  = [50] , alpha = 5.0 , random_state =0)
	nnclf.fit(x_train , y_train)
	print (" neural_network classifier scores for %s" % activation_func)
	print (nnclf.score(x_train , y_train) )
	print (nnclf.score(x_test , y_test) )


## working on real world dataset 

cancer_data =  load_breast_cancer()
x , y = cancer_data.data , cancer_data.target
x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 0)

scaler = MinMaxScaler()
## pre processing of data

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

nnclf =  MLPClassifier(hidden_layer_sizes = [50,50] , random_state = 0 , alpha = 5.0 )
nnclf.fit(x_train_scaled , y_train)
print ("the score on test data on Cancer data %f " % (nnclf.score(x_train_scaled , y_train)))
print ("the score for train data on Cancer  data %f "%(nnclf.score(x_test_scaled , y_test)))


## using regression in nueral_networks

from sklearn.neural_network import MLPRegressor

X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
fig, subaxes = plt.subplots(2, 3, figsize=(11,8), dpi=70)

X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

for thisaxisrow, thisactivation in zip(subaxes, ['tanh', 'relu']):
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
        mlpreg = MLPRegressor(hidden_layer_sizes = [100,100],
                             activation = thisactivation,
                             alpha = thisalpha,
                             solver = 'lbfgs').fit(X_train, y_train)
        y_predict_output = mlpreg.predict(X_predict_input)
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output,
                     '^', markersize = 10)
        thisaxis.plot(X_train, y_train, 'o')
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('MLP regression\nalpha={}, activation={})'.format(thisalpha, thisactivation))
     	plt.tight_layout()           
     	plt.show()               