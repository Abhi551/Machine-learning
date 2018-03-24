from sklearn.linear_model import Lasso 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import load_crime_dataset

import numpy as np 

## train_test_split 
x_crime , y_crime = load_crime_dataset()
x_train , x_test , y_train , y_test = train_test_split(x_crime , y_crime , test_size = .25 , random_state = 0)


## scalling factor
scaler =  MinMaxScaler()
x_scaled_train  = scaler.fit_transform(x_train)
x_scaled_test  =  scaler.transform(x_test)

lasso = Lasso(alpha = 2 , max_iter = 10000).fit(x_scaled_train , y_train)
print ("the lasso intercept_ is given by %.5f " %lasso.intercept_)
print ("the lasso coeffecients are ")
print (lasso.coef_)
print ("the number of zero coeffecients are %d " % np.sum( lasso.coef_ == 0))
print ("number of non zeros features or coeffecients %d " % np.sum( lasso.coef_ != 0 ))

print ("the score of Lasso on training data  %.5f "% lasso.score(x_scaled_train , y_train))
print ("the score of Lasso on test data is %.5f "%lasso.score(x_scaled_test , y_test))


## taking the values in a wide range


alpha_values = [0.5, 1, 2, 3, 5, 10, 20, 50]

for alpha in alpha_values :
	lasso = Lasso(alpha = alpha , max_iter = 10000).fit(x_scaled_train , y_train)
	print ("the score of Lasso on training data for alpha = %d is %.5f "% (alpha , lasso.score(x_scaled_train , y_train)))
	print ("the score of Lasso on test data for alpha = %d is %.5f "%(alpha , lasso.score(x_scaled_test , y_test)))
