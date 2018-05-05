from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import tree



iris_data = load_iris()
#print (iris_data.data)
x_iris , y_iris = iris_data['data'] , iris_data['target']
print (x_iris.shape)
print (y_iris.shape)

x_train , x_test , y_train , y_test =  train_test_split( x_iris , y_iris , test_size = .25 , random_state = 0)
print (x_train.shape)
print (y_train.shape)

DT_clf =  DecisionTreeClassifier()

clf = DT_clf.fit( x_train , y_train )

print ("score for training data " , DT_clf.score( x_train , y_train ))
print (" score for test data " , DT_clf.score( x_test , y_test))

#tree.export_graphviz(clf , out_file = "tree.dot")

## using prunning for preventing the overfitting of the data

x_iris , y_iris = iris_data['data'] , iris_data['target']

x_train , x_test , y_train , y_test = train_test_split( x_iris , y_iris , test_size = .25 , random_state = 0)

DT_clf = DecisionTreeClassifier ( max_depth = 3)

clf = DT_clf.fit ( x_train , y_train)
print ("\n\n")
print ("score for training data " , DT_clf.score( x_train , y_train ))
print (" score for test data " , DT_clf.score( x_test , y_test))


## on real world dataset

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


load_data = load_breast_cancer()
print (load_data['data'].shape)
print (load_data['target'].shape)

x_cancer = load_data['data']
y_cancer = load_data['target']


x_train , x_test , y_train , y_test = train_test_split( x_cancer , y_cancer , random_state = 0 , test_size = .25)

yes = 1
while yes == 1:

	max_depth = int(input("max_depth is = "))
	DT_clf = DecisionTreeClassifier( max_depth = max_depth , min_samples_leaf = 8 )

	clf = DT_clf.fit( x_train , y_train )

	print ("\n\n")
	print ("score for training data " , DT_clf.score( x_train , y_train ))
	print (" score for test data " , DT_clf.score( x_test , y_test))
	yes = int(input("do you want see more results = "))

export_graphviz(clf , out_file = "new.dot")

