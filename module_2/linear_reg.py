import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as  np 
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


X_r1 , Y_r1 = make_regression(n_samples = 100 , n_features = 1, n_informative = 1
							 , bias = 150 , noise = 30 , random_state = 0)

linear_reg = LinearRegression()
x_train , x_test , y_train , y_test = train_test_split( X_r1 , Y_r1 , test_size = .25 ,  random_state = 0)
print (x_test.shape)

linear_model = linear_reg.fit(x_train , y_train)

## linear_model can have multiple coeffecients depending on the features of the model

print ("the intercept for the linear model is %.5f" %(linear_model.intercept_))
print ("the slope for the linear model is %.5f "%(linear_model.coef_[0]))


## score for the linear_model

print ("the score of linear model on test data is %.5f " %linear_model.score(x_test , y_test))
print ("the score of linear model on train data is %.5f  "%linear_model.score(x_train ,  y_train))

## regression models for the same dataset 
## checking the best results we got k = 6 with highest score
## k_values = [1,3,5,6,7,9]
x_train , x_test , y_train , y_test = train_test_split(X_r1 , Y_r1 , test_size = .25 , random_state = 0)

# using a general value k =5
knn_regr = KNeighborsRegressor(n_neighbors = 5)
knn_regr.fit( x_train , y_train)
print ("the score for the knn regression on test data is %.5f"%knn_regr.score(x_test , y_test))
print ("the score for knn the regression on train data is %.5f"%knn_regr.score(x_train , y_train))


## plotting the linear model for the datasets
style.use('fivethirtyeight') 
plt.scatter(x_train , y_train , marker = 'o' , s = 45)
#print (linear_model.coef_*(x_test)+linear_model.intercept_)
plt.plot(x_test , linear_model.coef_*(x_test) + linear_model.intercept_ , color = "r")
plt.show()

