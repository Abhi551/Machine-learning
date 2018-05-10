## Multi Dimensional Scaling algorithm for reducing the dimension of data
import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt 

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler 
from sklearn.manifold import MDS 
from adspy_shared_utilities import plot_labelled_scatter


df = pd.read_csv('fruit_data_with_colors.txt', delimiter ="\t")

x_fruits = df[["mass" , "width" , "height" , "color_score"]]
y_fruits = df[["fruit_label"]] - 1


## preprocessing of data 
scaler = StandardScaler()

## making mean = 0 and variance = 1
x_fruits = scaler.fit_transform(x_fruits)
for i in range(2,5):

	mds = MDS( n_components = i)
	x_fruits_mds = mds.fit_transform(x_fruits)
	plot_labelled_scatter(x_fruits_mds , y_fruits , ["apple" , "mandrian" , "orange" , "lemon"])

## real world data for MDS (multi dimensional scaling)

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

cancer_data = load_breast_cancer()
x_data , y_data = cancer_data.data , cancer_data.target

## StandardScaing of the data

scaler = StandardScaler()
## fit and transform
x_data =  scaler.fit_transform(x_data)

## using M
mds = MDS(n_components = 2)

X_mds = mds.fit_transform(x_data)

plot_labelled_scatter(X_mds , y_data , ["malignant" , "benign"])
