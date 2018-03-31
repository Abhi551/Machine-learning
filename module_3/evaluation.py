import numpy as np
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

dataset = load_digits()

#print (dataset)

x , y = dataset.data , dataset.target
#print (x , y)

## creating data imbalance 

y_imb =  y.copy()
y_imb[y_imb != 1] = 0

#print (y_imb[:20])

x_train , x_test , y_train , y_test = train_test_split( x , y_imb , test_size = .25 , random_state = 0 )
print (np.bincount(y_imb))


## applying Decision Tree classifier on dataset

clf_dt =  DecisionTreeClassifier(max_depth = 3 )
clf_dt.fit(x_train , y_train)
y_predict_dt =  clf_dt.predict(x_test )
#print (clf_dt.score(x_test , y_test))

## confusion matix 
print (confusion_matrix(y_test , y_predict_dt))

## evaluating the Decision Tree
print ("accuracy score is for Decision Tree ")
print (accuracy_score( y_test , y_predict_dt ))
print ("precision score is for  Decision Tree ")
print (precision_score( y_test , y_predict_dt ))
print ("recall score is for Decision Tree " )
print (recall_score( y_test , y_predict_dt ))
print ("f1 score is for Decision Tree ")
print (f1_score( y_test , y_predict_dt))
print ("\n")
print ("classification_report of the Decision Tree is ")
print (classification_report(y_test , y_predict_dt , target_names = ['not 1' , '1']))
## applying Dummy Classifier dataset

print ("\n\n")
dummy = DummyClassifier( strategy = 'most_frequent')
dummy.fit(x_train , y_train)
y_predict_dummy = dummy.predict(x_test)
#print (y_predict_dummy)

## confusion matrix for the dummy classifier
print ("confusion matrix for the dummy classifier")
print (confusion_matrix(y_test , y_predict_dummy))

## logistic regression 

clf_logreg = LogisticRegression( C = 1 )
clf_logreg.fit( x_train , y_train)
y_predict_reg = clf_logreg.predict( x_test)

print ("\n\n")
## evaluating the logistic regression
print ("accuracy score is for regression ")
print (accuracy_score( y_test , y_predict_reg ))
print ("precision score is for regression ")
print (precision_score( y_test , y_predict_reg ))
print ("recall score is for regression ")
print (recall_score( y_test , y_predict_reg))
print ("f1 score is for regression ")
print (f1_score(y_test , y_predict_reg))
print ("classification_report for regression is")
print (classification_report( y_test , y_predict_reg , target_names = ['not 1 ' , '1']))

## SVM 

clf_svm = SVC(kernel = "linear" , C = 1)
clf_svm.fit(x_train , y_train)
y_predict_svm = clf_svm.predict(x_test)

print ("\n\n")

print ("accuracy score is for SVM ")
print (accuracy_score(y_test , y_predict_svm))
print ("precision score is for SVM ")
print (precision_score(y_test , y_predict_svm))
print ("recall score is for SVM ")
print (recall_score(y_test , y_predict_svm))
print ("f1 score is for SVM ")
print (f1_score(y_test , y_predict_svm))
print ("\n")
print ("classification report for SVM is")
print (classification_report(y_test , y_predict_svm , target_names = ['not 1' , '1']))
