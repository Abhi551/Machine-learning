import numpy as np 
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification , make_blobs 
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer 

cmap_bold = ListedColormap(["#FFFF00" ,  "#00FF00" , "#000FF" , "000000"])

## fruits datasets

fruits = pd.read_csv("fruit_data_with_colors.txt" , delimiter = "\t")

#print fruits.head()

## selecting the impoprtant features
feature_names_fruits = ["height" , "width" , "mass" ,  "color_score" ]
## labeling the fruits 

## giving the x_features and y_features

x_fruits = fruits[feature_names_fruits]
y_fruits = fruits["fruit_label"]

## using only 2 features  

x_fruits_2d = fruits[["height" , "width"]]
y_fruits_2d = fruits["fruit_label"]


## making synthetic datasets  for simple regression
from sklearn.datasets import make_regression

x_r1 , y_r1 = make_regression(n_samples= 100 , n_features = 1 , n_informative = 1 , 
			 bias = 150 ,  noise = 42  , random_state = 0)

plt.scatter(x_r1 , y_r1 , marker = "o" , s = 50)
plt.title("Sample regression problem for one input variable")
plt.show()

## making synthetics datasets for complex regression

from sklearn.datasets import make_friedman1

x_f1 , y_f1 = make_friedman1(n_samples = 150 , n_features = 7 ,
							random_state = 0)
#print (x_f1)
#print (x_f1[:,1])
plt.title("complex regression problem with one input variable")
plt.scatter(x_f1[: , 2] , y_f1 , marker = 'o' , s = 50 )
plt.show()

## synthetic data for classification (binary) with 2 features

x_c1 , y_c1 = make_classification(n_samples = 150 , n_features  = 2 , n_redundant = 0 ,
				n_informative =  2 , n_clusters_per_class = 1 , flip_y = 0.1 ,
				class_sep = .5 , random_state = 0 )
plt.title("binary classification with 2 features")
plt.scatter(x_c1[: , 0] , x_c1[: , 1] , marker = "o" , c = y_c1 , s = 50 )
plt.show()

