from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 

X_r1 , Y_r1 = make_regression(n_samples = 100 , n_features = 1 , n_informative = 1 , bias = 150.0 ,
			noise = 30 , random_state = 0)

style.use('ggplot')
plt.scatter(X_r1 ,  Y_r1 , marker = 'o' , color = 'c' , s = 45)
plt.title('simple make_regression ')
plt.show()

## train_test_split 
def regression_model():
	
	from sklearn.model_selection import train_test_split 
	X_train , X_test , Y_train , Y_test  = train_test_split(X_r1 , Y_r1 , test_size = .25)
	print (X_train.shape)


	## using the regression model 
	knn_reg = KNeighborsRegressor(n_neighbors = 5)
	knn_reg.fit(X_train , Y_train )
	print ("the r squared test score is   {0:.4f}".format(knn_reg.score(X_test , Y_test)))
	print ("the r squared train score is {0:.4f}".format(knn_reg.score(X_train , Y_train)))

	## predicting the values in test set 
	print (knn_reg.predict(X_test))


	## using plots to describe the behaviour 


	x_train , x_test , y_train , y_test = train_test_split(X_r1[0 :: 5] , Y_r1[0 :: 5])
	x_input =  np.linspace(-3 , 3 , 50)

	print ("previous shape of input data %s" % str(x_input.shape))
	## changing the input to similar shape as that of training set
	print ("shape of train data %s " %str(x_train.shape))
	## in case number of rows are unknown we use -1 and 1 column
	x_input = x_input.reshape(-1,1)
	print ('changed shape of input data %s ' % str(x_input.shape))
	print ('shape of train data  %s ' % str(x_train.shape))



	for k in [3 , 5 ]:

		knn_reg = KNeighborsRegressor(n_neighbors = k)
		knn_reg.fit(x_train , y_train)
		y_output = knn_reg.predict(x_input)
		print (y_output.shape)
		if k == 3:

			label = knn_reg.score(x_train , y_train)
			print ("score of train data " , label2)
			plt.scatter(x_train , y_train , marker = 'o' , color = 'green' , s = 45 , label = label )
			plt.scatter(x_input , y_output , marker = '^' ,  color = 'red' , s = 35  )
			plt.legend()
			plt.title(" k = 3 regression ")
			plt.show()
		if k == 5 :
			label =  knn_reg.score(x_train , y_train)
			plt.scatter(x_train ,  y_train , marker = 'o' , color = 'green' , s =45 , label = label)
			plt.scatter(x_input , y_output , marker = '^' , color = 'red' , s = 30 )
			plt.title("k = 5 regression")
			plt.legend()
			plt.show()
	

#regression_model()

### Regression model complexity as function of K

from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
x_r1 , y_r1  = make_regression(n_samples = 100 , n_features = 1  , n_informative = 1 ,  bias = 200 ,
								noise = 30 ,  random_state = 0 )
x_train , x_test , y_train , y_test = train_test_split( x_r1 , y_r1 , test_size = .25 , random_state = 0)
x_input = np.linspace(-3 , 3 , 500)
#print (x_train)
print ("shape of x_train %s " % str(x_train.shape))
print ("shape of x_input %s " % str(x_input.shape))
x_input = x_input.reshape(-1,1)
print ("new shape of x_input %s " % str(x_input.shape))

k_values = [1 , 3 , 7 , 9 , 11 , 55]
for k in k_values :
	knn_reg = KNeighborsRegressor(n_neighbors = k)
	knn_reg.fit(x_train , y_train)
	y_output = knn_reg.predict(x_input)
	print (knn_reg.score(x_train , y_train))
	plt.plot(x_train , y_train , 'o' ,alpha = 1 , color = 'green' , label =  "training score %.4f"%(knn_reg.score(x_train , y_train)))
	plt.plot(x_test , y_test , '^' , alpha = .8 , color = "red" , label = "test score %4f"%knn_reg.score(x_test , y_test))
	plt.plot(x_input , y_output , label = "predicted data")
	plt.title('k =  %s'%k)
	plt.legend()
	plt.show()

