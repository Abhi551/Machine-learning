from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 


fruits_pd = pd.read_csv('fruit_data_with_colors.txt' , delimiter = '\t')
df_fruits = pd.DataFrame(fruits_pd)
print (df_fruits.head())

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
x_fruits_2d = df_fruits[['height' , 'width']]
y_fruits_2d = df_fruits['fruit_label']

## make into a binary problem: apples vs everything else

y_fruits_apple = y_fruits_2d == 1 
x_train, x_test, y_train, y_test = train_test_split(x_fruits_2d.as_matrix() , y_fruits_apple.as_matrix() , random_state = 0)


clf_reg = LogisticRegression(C = 100)

coef = clf_reg.fit(x_train ,  y_train)
print ("\n\n")
print ("the intercept of the Logistic Regression is %s " % coef.intercept_)
print ("the coeffecients of the Logistic Regression are %s"  %coef.coef_)


print ("\n\n")
print ("prediction score in C = 100 for training data %.5f" % clf_reg.score(x_train , y_train))
print ("prediction score in C = 100 for test data %.5f" % clf_reg.score(x_test , y_test))


## for different values of C , change in score

clf_reg = LogisticRegression(C = 1)
coef = clf_reg.fit(x_train , y_train)
print ("\n\n")
print ("prediction score in C = 1 for training data %.4f" % coef.score(x_train , y_train))
print ("prediction score in C = 1 for test data %.4f" % coef.score(x_test , y_test))

clf_reg = LogisticRegression(C = 350)
coef =clf_reg.fit(x_train , y_train)

print ("\n\n")
print ("prediction score in C = 350 for training data %.4f" % coef.score(x_train , y_train))
print ("prediction score in C = 350 for test data %.4f" % coef.score(x_test, y_test))

## applying the logistic regression on synthetic data


x_c1 , y_c1 = make_classification(n_samples = 100 , n_features = 2 ,n_redundant = 0 , n_informative = 2 ,
			n_clusters_per_class = 1 , flip_y = .1  , class_sep = .5 , random_state =0)

x_train , x_test , y_train , y_test = train_test_split(x_c1 , y_c1 , random_state = 0)

clf_reg = LogisticRegression(C = 100)
clf_reg.fit(x_train , y_train)

print ("\n\n")
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf_reg.score(x_train , y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf_reg.score(x_test , y_test)))

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x_cancer , y_cancer = load_breast_cancer(return_X_y = True)

x_train , x_test , y_train , y_test =train_test_split(x_cancer , y_cancer , test_size = .25)
clf_reg = LogisticRegression(C = 100)
clf_reg.fit(x_train , y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf_reg.score(x_train , y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf_reg.score(x_test ,y_test)))
