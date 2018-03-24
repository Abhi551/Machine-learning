import matplotlib.pyplot as plt 
import numpy as np 
from adspy_shared_utilities import load_crime_dataset
from sklearn.linear_model import Ridge
from sklearn.model_selection  import train_test_split


x_crime , y_crime = load_crime_dataset()
x_train , x_test , y_train , y_test = train_test_split(x_crime , y_crime , test_size = .25 , random_state = 0)


## the regularsiation factor

lin_ridge = Ridge(alpha = 20)

coeffecients = lin_ridge.fit(x_train , y_train)
print ("intercept for the ridge regression is %.5f " %coeffecients.intercept_)

coeff = coeffecients.coef_
print ("other coeffecients for the ridge reression are ")
#print (coeff)

## score of the ridge regression on training data and test data
print ("the score of Ridge on training data  %.5f "% lin_ridge.score(x_train , y_train))
print ("the score of Ridge on test data is %.5f "%lin_ridge.score(x_test , y_test))


## Ridge Regression with feature normalisation

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()

## split the data 
x_train , x_test , y_train , y_test = train_test_split(x_crime , y_crime , random_state = 0 , test_size = .25)
print (x_train.shape)

## min and max of the feature using scaling 
## applying fit_transfom to apply both operations at once 

x_scaled_train = scaler.fit_transform(x_train)

## using same set of parameters apply transformation on x_test

x_scaled_test = scaler.transform(x_test)


## now apply ridge regression

lin_ridge =  Ridge(alpha = 20 )

## fitting the data
lin_ridge.fit(x_scaled_train , y_train)
## the resutls are better for the regularised data
## checking the score for training and test data
print ("this is the score of ridge regression for scaled test data %.5f" % lin_ridge.score(x_scaled_test , y_test))
print ("this is the score for ridge regression for scaled training data %.5f" % lin_ridge.score(x_scaled_train , y_train))


## using regularisation for different values of alpha

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train , x_test , y_train , y_test = train_test_split(x_crime , y_crime)

## scaling the data
x_scaled_train = scaler.fit_transform(x_train)
x_scaled_test = scaler.transform(x_test)
aplhas = list(range(1,82,10))
for alpha in aplhas :
	ridge =  Ridge(alpha = alpha)
	coeffecients =  ridge.fit(x_scaled_train , y_train)
	print ("the score for training data for k = %d is %.6f " %(alpha , ridge.score(x_scaled_train , y_train)))
	print ("the score for test data for k = %d is %.6f " %(alpha , ridge.score(x_scaled_test , y_test)))
	
