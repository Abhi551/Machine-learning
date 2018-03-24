import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

try : 
	import seaborn 
except ImportError :
	pass

## loading datasets 

## 1. Fruits_data.csv
## to set precision after the decimal value 
np.set_printoptions(precision = 2)

## new way to read a csv file using read_table
## but we will use pd.read_csv()
new_df = pd.read_table('fruits.csv' , delimiter =',')
new_df = pd.DataFrame(new_df)
#print (new_df.head())


## reading the csv file 
fruits_data = pd.read_csv('fruits.csv')
## converting the data in dataframe 
df_fruits = pd.DataFrame(fruits_data)
print (df_fruits.head())

## classifying the data into X and Y i.e. label the features and Target values
X_data = df_fruits[['mass' , 'height' , 'width' , 'color_score']]
Y_data = df_fruits['fruit_label']
print ('\nX  features \n' , X_data.head())
print ('\ntarget values \n' , Y_data.head())

## train_test_split 
X_train , X_test , Y_train , Y_test = train_test_split(X_data , Y_data , random_state = 0 , test_size = .25)
print (X_train.shape)
print (X_test.shape)



## we can do it without scaling also

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train , Y_train)
print (knn.score(X_test , Y_test))

## using the scaling factor instead of directly using the data set

scaler = MinMaxScaler()
## fit and transform on training set 
scaled_X_train = scaler.fit_transform(X_train)
## using same parameters on test set
scaled_X_test = scaler.transform(X_test)


## applying knn classifier on scaled data
knn =KNeighborsClassifier(n_neighbors = 5)
knn.fit(scaled_X_train , Y_train)
print (knn.score(scaled_X_train , Y_train))
print (knn.score(scaled_X_test , Y_test))

test = [[5.5, 2.2, 10, 0.70]]
print (knn.predict(test)[0])


## creating synthetic datasets for illustration

from matplotlib.colors import ListedColormap
from matplotlib import style
from sklearn.datasets import load_breast_cancer


## synthetic dataset fot simple regression

from sklearn.datasets import make_regression

style.use('fivethirtyeight')
X_r1 , Y_r1 = make_regression(n_samples = 100 , n_features = 1 , n_informative = 1,
			bias = 150 , noise = 30 , random_state = 0 )

#print ("X_r1 \n" , X_r1 , "\nY_r1\n" , Y_r1)

plt.scatter(X_r1 , Y_r1 , marker = 'o' , color = 'c')
plt.title("simple regression")
plt.show()

## synthetic data for more complex regression
from sklearn.datasets import make_friedman1

X_f1 , Y_f1 = make_friedman1(n_samples = 100 , n_features = 7 , random_state = 0)
#print (X_f1 ,  X_f1.shape , type(X_f1))

plt.scatter(X_f1[:,0] , Y_f1 , color = 'r')
plt.scatter(X_f1[:,1] , Y_f1 , color = 'c')
plt.title("complex regression datasets")
plt.show()

## synthetic data for classification(binary)

from sklearn.datasets import make_classification 

X_c1 , Y_c1 =  make_classification(n_samples = 100 , n_features = 2 , n_redundant = 0 , 
				n_informative = 2 , n_clusters_per_class = 1 , flip_y = .1 ,
				class_sep = .5 , random_state = 0)

plt.title("binary classification")
plt.scatter(X_c1[: , 0] , X_c1[: , 1] , c = Y_c1 ,  marker = 'o' )
plt.show()


from sklearn.datasets import make_blobs

## more difficult synthetic datasets for binary classification
## with classes that are not linearly seperable

X_b1 , Y_b1 = make_blobs(n_samples = 100 , n_features = 2 , centers = 8 ,
						cluster_std = 1.3 ,  random_state = 4)
Y_b1 = Y_c1%2
plt.scatter(X_b1[ : , 0] ,  X_b1[ : ,1] , c = Y_b1 , marker = 'o' )
plt.show()


## Breast cancer dataset for classification
from sklearn.datasets import load_breast_cancer
X , Y =  load_breast_cancer(return_X_y = 1)
#print (X , Y)

