import numpy as np 
from sklearn.datasets  import load_digits
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier

dataset = load_digits()

x , y  = dataset.data , dataset.target 
print (dataset.target)
print (np.bincount(y))
y_imbalance = y.copy()

y_imbalance[y_imbalance != 1 ] = 0
print (np.bincount(y_imbalance))

x_train , x_test , y_train , y_test = train_test_split(x , y_imbalance , test_size = .25 , random_state = 0)

dummy = DummyClassifier( strategy = "most_frequent")
dummy.fit(x_train , y_train)

y_predict = dummy.predict(x_test)

confusion = confusion_matrix(y_test , y_predict)
#print (y_test.shape)
print (confusion)

## choosing different classifier for confusion matrix

##2. stratified classifier 

from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

dummy = DummyClassifier( strategy = 'stratified')
x_train , x_test , y_train , y_test = train_test_split(x , y_imbalance , test_size = .25 ,random_state = 0)

dummy.fit(x_train , y_train)
y_predict = dummy.predict(x_test)

print (confusion_matrix(y_test , y_predict))


## using confusion matrix on different models
## SVM
from sklearn.svm import SVC 

clf_svm = SVC( kernel = "linear" , C = 1)
clf_svm.fit(x_train , y_train)
y_predict = clf_svm.predict(x_test)
print (confusion_matrix( y_test , y_predict ))

## logistic regresion
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()
clf_lr.fit(x_train , y_train)

y_predict = clf_lr.predict(x_test)
print (confusion_matrix( y_test , y_predict))

## decision tree
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier( max_depth = 3 )
clf_dt.fit(x_train , y_train)
y_predict = clf_dt.predict(x_test)
print (confusion_matrix(y_test , y_predict))
