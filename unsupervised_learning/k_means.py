from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from adspy_shared_utilities import plot_labelled_scatter
from sklearn.preprocessing import MinMaxScaler

import pandas as pd 

x , y = make_blobs(random_state = 10 )

kmeans =  KMeans(n_clusters = 3)
kmeans.fit(x)

plot_labelled_scatter(x , kmeans.labels_ , ["1" , "2" , "3"])

## fruits data for clustering 

fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']].as_matrix()
y_fruits = fruits['fruit_label'] - 1

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X_fruits)

plot_labelled_scatter(X_fruits , kmeans.labels_ , [1,2,3,4])

## using pre processing on data

scaler = MinMaxScaler()
X_fruits = scaler.fit_transform(X_fruits)

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X_fruits)

## plot
plot_labelled_scatter(X_fruits , kmeans.labels_ , [1,2,3,4])