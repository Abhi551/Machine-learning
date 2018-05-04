import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor 
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error , r2_score
from matplotlib import style

dataset = load_diabetes()
#print (dataset)

print (dataset.data.shape)
x , y = dataset.data[ : , None , 6] , dataset.target

x_train , x_test , y_train , y_test =  train_test_split( x , y , test_size = .25 , random_state = 0)

lr = LinearRegression()
coef = lr.fit( x_train , y_train )

print (coef.intercept_)
print (coef.coef_)

y_predict_lr = lr.predict(x_test)

## training the dummy regressor 

dummy =  DummyRegressor( strategy =  'mean' )
dummy.fit( x_train , y_train )

y_predict_dummy = dummy.predict( x_test )

print ("r2 score for dummy is %.2f" % (r2_score( y_test , y_predict_dummy )))
print ("mean_squared_error for dummy is %.3f" %(mean_squared_error( y_test , y_predict_dummy )))

## for linear regressor 


print ("r2 score for linear model is %.2f" % (r2_score( y_test , y_predict_lr )))
print ("mean_squared_error for linear model  is %.3f" %(mean_squared_error( y_test , y_predict_lr )))

## plots
style.use("ggplot")
plt.scatter( x_test , y_test , color = 'k')
plt.plot( x_test , y_predict_lr , color = 'g')
plt.plot( x_test , y_predict_dummy , color = 'r')
plt.show()