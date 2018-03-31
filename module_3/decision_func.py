from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style

dataset = load_digits()
#print (dataset)
x_data , y_data = dataset.data , dataset.target
## creating an imbalance dataset

y_imb =  y_data.copy()
y_imb[y_imb != 1] = 0

x_train , x_test , y_train , y_test = train_test_split(x_data , y_imb)

## fitting the data
lr = LogisticRegression()
lr.fit(x_train , y_train)
## applying the decision function to LogisticRegression
y_scores = lr.decision_function(x_test)
#print (y_scores[:20])

## combining it with the result or y_targets
dec_func = list(zip(y_test[:20] , y_scores[:20]))
#print (dec_func)
print ("decision function is")
for i in range(20):
	print (dec_func[i])
lr =  LogisticRegression()
lr.fit(x_train , y_train)

y_prob = lr.predict_proba(x_test)
#print (y_prob[:20 , 1])
## zipping it with y_test

prob_func = list(zip(y_test[:20] , y_prob[:20 , 1]))
print ("\n\npredict prob_func is")
#print (prob_func)
for i in range(20):
	print (prob_func[i])

style.use('ggplot')
plt.figure()

precision , recall , thresholds = precision_recall_curve(y_test , y_scores)
#print (recall , precision ,thresholds)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.xlim([0.0 , 1.01])
plt.ylim([0.0 , 1.01])
plt.plot(precision , recall ,  label = "Precision Recall Curve")
plt.plot(closest_zero , closest_zero_p , closest_zero_r , 'o' , markersize = 12 , c = 'r')
#plt.legend()
plt.xlabel("precision")
plt.ylabel("recall")
plt.show()

from sklearn.metrics import roc_curve , auc
x_train , x_test , y_train , y_test = train_test_split (x_data , y_imb)

clf_lr = LogisticRegression()
clf_lr.fit(x_train , y_train)
y_score_lr = clf_lr.decision_function(x_test)

## roc curve gives the plotting point for roc
fpr_lr , tpr_lr , _ = roc_curve( y_test , y_score_lr)
auc_lr = auc(fpr_lr , tpr_lr)
plt.figure()
plt.xlim([-.01 , 1.00])
plt.ylim([-.01 , 1.00])
plt.plot(fpr_lr , tpr_lr , lw = 3 , label = "LogisticRegression area {:.2f}".format(auc_lr))
plt.xlabel("False Positive Rate")
plt.ylabel("True  Positive Rate")
plt.title("ROC curve ")
plt.legend(loc = "lower right" , fontsize = 13)
plt.plot([0 , 1] , [0 , 1] , color = 'navy' , lw = 3 , linestyle = '--')
plt.show()