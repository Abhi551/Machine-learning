import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_friedman1

x_f1 , y_f1 = make_friedman1(n_samples = 100 , n_features = 7 , random_state =0)

x_train , x_test , y_train , y_test = train_test_split( x_f1 , y_f1 , random_state = 0)

## using linear regression on the data
lin_reg =  LinearRegression()
coeffecients = lin_reg.fit(x_train , y_train)
print ("intercept of the linear regression model = " , coeffecients.intercept_)
print ("coeffecients of the linear regression model = " , coeffecients.coef_)
print ("\n\n")
print ("linear regression score on training data" , lin_reg.score(x_train , y_train))
print ("linear regression score on test data ", lin_reg.score(x_test , y_test))


## using polynomial regression without using any regularization parameter
## i.e. using it with Ridge or Lasso

from sklearn.datasets import make_friedman1


x_f1 , y_f1 = make_friedman1(n_samples = 100 , n_features = 7 , random_state = 0)

## transform the original input data to add polynomial features upto 2 degrees
poly = PolynomialFeatures( degree = 2)

## fitting the polynomial
x_f1_poly = poly.fit_transform(x_f1)

x_train , x_test , y_train , y_test = train_test_split(x_f1_poly ,y_f1 , random_state = 0)
#print (x_train.shape ,x_f1.shape)

lin_reg = LinearRegression()
## fitting the polynomial with degree = 2
coeffecients = lin_reg.fit(x_train , y_train)
print ("\n\n")
print ("the intercept is " , coeffecients.intercept_)
print ("the coeffecients are " , coeffecients.coef_)
print ("\n\n")
print ("the score for  polynomial regression on training set is =",lin_reg.score(x_train , y_train))
print ("the score for polynomial regression on test set is =" , lin_reg.score(x_test , y_test))


## adding regularization parameter in polynomial regression Ridge 

from sklearn.datasets import make_friedman1 
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

## datasets 
x_f1 , y_f1 = make_friedman1(n_samples = 100 , n_features = 7 , random_state = 0)

poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x_f1)

## splitting the data

x_train , x_test , y_train , y_test = train_test_split(x_poly , y_f1 , random_state = 0)
 	
alpha_values = [1 , 2 , 5 , 8]
	## using ridge regression
for alpha in alpha_values:
	ridge_reg =  Ridge(alpha = alpha)
	coeffecients = ridge_reg.fit(x_train , y_train)

	## the coeffecients and intercept 
	print ("APLHA = " , alpha)
	print ("\n\n")
	print ("the coeffecients of the ridge regression are = " , coeffecients.coef_)
	print ("the intercept of the ridge regression is =  " , coeffecients.intercept_)

	print ("the score on training data " , ridge_reg.score(x_train , y_train))
	print ("the score on test  data " ,  ridge_reg.score(x_test , y_test))


## alpha = 1 gives the best results