from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from matplotlib import style
import numpy as np

fruits = pd.read_csv('fruit_data_with_colors.txt' , delimiter = "\t")
df_fruits = pd.DataFrame(fruits)
#print (df_fruits.head())

x_fruits = df_fruits[[ 'width' , 'height' ]]
y_fruits = df_fruits['fruit_label']

x_train , x_test , y_train , y_test = train_test_split(x_fruits , y_fruits , test_size = .25)


clf = LinearSVC( C = 4 )
clf.fit(x_train , y_train)
print (clf.score(x_test , y_test) )
print (clf.score(x_train , y_train))
print ("\n\n")
print ("intercept of the classifier is " , clf.intercept_)
print ("coeffecients of the classifier is" , clf.coef_)

style.use('ggplot')
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])
plt.scatter(x_fruits[['height']] ,  x_fruits[['width']] , c = y_fruits , cmap = cmap_fruits, alpha = .7 , s = 65)


x_range = np.linspace(-10 , 15)
print (x_range)
for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_range, -(x_range * w[0] + b) / w[1], c=color, alpha=.8)
    plt.xlim(-2 , 12)
    plt.ylim(-2 , 15)
plt.show()