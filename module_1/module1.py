import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np 
import pandas as pd 
from matplotlib import style
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D

#style.use('ggplot')

## reading the csv file by pd.read_csv
df_fruits = pd.read_csv('fruits.csv')

print (df_fruits.head())

X =  df_fruits[['mass' , 'height' , 'width']]
Y = df_fruits.fruit_label
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , random_state = 0)

## 3d plotting 
fig = plt.figure()
ax  = fig.add_subplot(111 , projection = '3d')
ax.scatter(X_train['mass'] , X_train['height'] , X_train['width'] , c = Y_train)
plt.show()
## using feature pair plot 
cmap = cm.get_cmap('gnuplot')



scatter = pd.scatter_matrix(X_train , c = Y_train , s = 40  , figsize = (9,9) , alpha = .9 ,cmap = cmap)
scatter1 = pd.scatter_matrix(X_train , c = Y_train , s = 52 , figsize = (11 , 11) , alpha = .95 )



#print (df_fruits.fruit_label.unique())
#print (df_fruits.fruit_name.unique())

## a dictionary to associate the fruits_label with the fruits_name

dict_fruit = dict(zip(df_fruits.fruit_label.unique(),df_fruits.fruit_name.unique()))
print (dict_fruit)




#print (df_fruits['mass'].head())
#print (df_fruits[['mass' , 'width']].head())


X = df_fruits[['mass' , 'width' , 'height' ]]
Y = df_fruits['fruit_label']


## train_test_split( X , Y , test_size , train_size , random_state , shuffle)
## where X,Y are arrays 
## train_size , test_size gives the size of traing and test data , between (0,1)

## since we don't have different training and test data  we split the data into 2 categories

X_train , X_test , Y_train , Y_test = train_test_split(X , Y  , test_size = .7, random_state = 0)

#print (X_train.shape,X_test.shape)
#print (df_fruits.shape)

## classifier object 
knn = KNeighborsClassifier(n_neighbors = 2)

## fitting the training data sets
knn.fit(X_train , Y_train)
print (knn.fit(X_train , Y_train))

## estimating the accuracy of the future data
print (knn.score(X_test , Y_test))

## predicting the fruit using random data 
fruit_prediction = knn.predict([[173 , 5.5 , 6.5]])

print (dict_fruit[fruit_prediction[0]])
	
## checking the accuracy of knn classifier with value of k

k_range = range(1,15)
scores = []

for k in k_range:
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(X_train , Y_train)
	#print (knn.score(X_test , Y_test))
	scores.append(knn.score(X_test , Y_test))

plt.figure()
plt.scatter(list(k_range) , scores)
plt.xlabel('k_range')
plt.ylabel('scores')
plt.show()

