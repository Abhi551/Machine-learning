## unsupervised model used for extracting important variable from large set of variables
## in a data set .
## It extracts low dimensinonal set of features from a high dimensinonal dataset
## to capture as much information as possible 
## best when 3 or more features are present in dataset

import numpy as  np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adspy_shared_utilities import plot_labelled_scatter

cancer_data = load_breast_cancer()

## returns data and target both using return_X_y
x_cancer ,  y_cancer = load_breast_cancer(return_X_y = True)

## performing preprocessing on the datasets
## so that each feature have zero mean and unit variance

scaler =  StandardScaler()
x_fit = scaler.fit(x_cancer)
x_transform = x_fit.transform(x_cancer)
print (x_transform.shape)
## the final results will give the data which  have zero mean and variance of data is unity          


## specify the PCA object with 2 features to retain only 
## and fitting the transformed data in PCA object

pca = PCA(n_components = 2).fit(x_transform)
print (pca)
## last step is to 
## put the transformed data in the pca object to give the final transformed data

x_final = pca.transform(x_transform)
print (x_final.shape)

## using the same result on real world datasets

plot_labelled_scatter(x_final , y_cancer , ['malignant', 'benign'])


## creating a heatmap for each feature
## i.e. plotting the magnitude of each feature value for first 2 principal components

fig =  plt.figure( figsize = (8,4) )
print (pca.components_.shape)
plt.imshow(pca.components_ , interpolation = 'none' , cmap = "plasma")

feature_names = list(cancer_data.feature_names)

plt.gca().set_xticks(np.arange(-.5 , len(feature_names)))
plt.gca().set_yticks(np.arange(.5 , 2 ))
plt.gca().set_xticklabels(feature_names , rotation = 90 , ha = "left" , fontsize = 12)
plt.gca().set_yticklabels(["First PC" , "Second PC"] , va = "bottom" , fontsize = 12)

plt.colorbar(orientation = "horizontal" , ticks = [pca.components_.min() , 0 ,
												 pca.components_.max()] , pad = .65) 

plt.show()

## on fruits dataset 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adspy_shared_utilities import plot_labelled_scatter

df = pd.read_csv('fruit_data_with_colors.txt', delimiter ="\t")

## preprocessing of data

x_fruits = df[['mass','width','height', 'color_score']]
y_fruits = df[['fruit_label']]

print (x_fruits.head())
scaler =  StandardScaler()
x_fruits = scaler.fit(x_fruits).transform(x_fruits)

## using PCA
for i in range(2,5):
	pca = PCA(n_components = 2).fit(x_fruits)
	x_pca = pca.transform(x_fruits)	
	plot_labelled_scatter(x_pca , y_fruits , ["apple" ,  "mandarian" , "orange" , "lemon"])

