from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from adspy_shared_utilities import plot_labelled_scatter
from sklearn.datasets import load_breast_cancer

import numpy as np 
import pandas as pd

## reading the fruits dataset
df = pd.read_csv('fruit_data_with_colors.txt', delimiter ="\t")
x_fruits , y_fruits =  df[["mass" , "color_score" , "height" , "width" ]] , df[["fruit_label"]] - 1

scaler = MinMaxScaler()
x_fruits =  scaler.fit_transform(x_fruits)

tsne =  TSNE(n_components  = 2)
x_tsne =  tsne.fit_transform(x_fruits)

plot_labelled_scatter(x_tsne , y_fruits ,['apple' , 'mandrian' , "orange" , "lemon"])


