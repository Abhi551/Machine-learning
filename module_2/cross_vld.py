## cross validation is used for evaluating the model not tune it 

import numpy as np 
import pandas as pd

print (np.logspace( 2 , 3 , num=10 ))
## by default num = 50
#print (np.logspace( 1 , 2 ))


## cross validation for fold = 5

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier


fruits =  pd.read_csv( 'fruit_data_with_colors.txt' , delimiter = "\t")
df_fruits = pd.DataFrame(fruits)

print ("\n\n")
print (df_fruits.head())
x_fruits = df_fruits[[ 'width' ,  'height' ]]
y_fruits = df_fruits['fruit_label']
#print (x_fruits.as_matrix())
clf = KNeighborsClassifier( n_neighbors = 5)
cv_scores = cross_val_score( clf , x_fruits , y_fruits , cv = 5)

print ("\n\n")
print ("scores for 5 fold",cv_scores)
print ("mean scores " , np.mean(cv_scores))



## Validation curve example
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC

param_range = np.logspace( -3 , 3 , 4)

train_scores , test_scores = validation_curve( SVC(kernel = 'linear') , x_fruits , y_fruits , param_name = 'gamma' , 
											  param_range = param_range ,  cv =3)
print( train_scores)
print (test_scores)




