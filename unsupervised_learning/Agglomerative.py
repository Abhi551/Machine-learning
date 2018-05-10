from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from adspy_shared_utilities import plot_labelled_scatter
from scipy.cluster.hierarchy import ward , dendrogram
import pandas as pd 
import matplotlib.pyplot as plt

X , y = make_blobs( random_state = 10)

cls =  AgglomerativeClustering(n_clusters = 3)
cls_new = cls.fit_predict(X)

plot_labelled_scatter(X , cls_new , [1,2,3])


## another sample for creating a dendrogram
## Agglomerative Algorithm works in similar down to up manner 
X , y = make_blobs(n_samples = 10 , random_state = 10)

plot_labelled_scatter(X , y , [1,2,3])

plt.figure()
dendrogram(ward(X))
plt.show()