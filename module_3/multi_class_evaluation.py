from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from matplotlib import style
from sklearn.metrics import confusion_matrix , classification_report , precision_score
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 

dataset = load_digits()
x , y = dataset.data , dataset.target

x_train , x_test , y_train , y_test = train_test_split( x , y)
clf_svm = SVC(kernel = "linear" )
clf_svm.fit(x_train , y_train)
y_predict = clf_svm.predict(x_test)
confusion_mat = confusion_matrix(y_test , y_predict )
print (confusion_mat)

## confusion matrix using visualization
df_cm = pd.DataFrame(confusion_mat , index = [i for i in range(10)] , columns = [j for j in range(10)])
plt.figure()
sns.heatmap(df_cm , annot = True)
print ("\n\nClassification report for linear kernel ")
print (classification_report(y_test , y_predict))
plt.title("SVM Linear Kernel ")
plt.ylabel("True values ")
plt.xlabel("Predicted values")
plt.show()

print ("\n\n")
## calculatin the micro and macro average
print ("Micro Score Average {:.2f}".format(precision_score(y_test , y_predict , average = 'micro')))
print ("Macro Score Average {:.2f} ".format(precision_score(y_test , y_predict , average = 'macro')))
## similarly applying the heatmap using different kernel

clf_svm = SVC(kernel = "rbf")
clf_svm.fit(x_train , y_train)
y_predict =  clf_svm.predict(x_test)
confusion_mat = confusion_matrix(y_test , y_predict)
print (confusion_mat)
print ("\n\n")
df_cm = pd.DataFrame(confusion_mat , index = [i for i in range(10)] , columns = [j for j in range(10)])
print ("Micro Score Average {:.2f}".format(precision_score(y_test , y_predict , average = "micro")))
print ("Macro Score Average {:.2f}".format(precision_score(y_test , y_predict , average = 'macro')))
plt.figure()
sns.heatmap(df_cm , annot = True)
print ("\n\nClassification report for rbf kernel")
print (classification_report(y_test , y_predict))
plt.title("SVM rbf Kernel")
plt.xlabel("True values")
plt.ylabel("Predicted")
plt.show()

## applying same SVM on polynomial kernel 
clf_svm = SVC(kernel = "poly")
clf_svm.fit(x_train , y_train)
y_predict =  clf_svm.predict(x_test)
confusion_mat = confusion_matrix(y_test , y_predict)
print (confusion_mat)

df_cm = pd.DataFrame(confusion_mat , index = [i for i in range(10) ] , columns = [j for j in range(10) ])
plt.figure()
sns.heatmap(df_cm , annot =True)
print ("\n\n")
print ("Micro Score Average {:.2f}".format(precision_score(y_test , y_predict , average = 'micro')))
print ("Macro Score Average {:.2f}".format(precision_score(y_test , y_predict , average = 'macro')))
print ("\n\nClassification report for polynomial kernel")
print (classification_report(y_test , y_predict))
plt.title("SVM with polynomial kernel")
plt.xlabel("True values ")
plt.ylabel("Predicted values")
plt.show()


