from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_blobs

## creating synthetics data

x_d , y_d = make_blobs( n_samples = 100 , n_features = 2 , centers = 8 ,
						cluster_std = 1.3 , random_state = 4)

## y_d = y_d%2

x_train , x_test , y_train , y_test =  train_test_split( x_d , y_d , test_size = .25)

## change in prediction score with the gamma parameter
gamma_values = [.001 , 1 , 5 ]
for gamma in gamma_values :
	clf_svm = SVC(kernel = 'rbf' , gamma = gamma  )
	clf_svm.fit(x_train , y_train)
	print (clf_svm.score( x_train , y_train ))
	print (clf_svm.score( x_test , y_test ))



## tuning the gamma parameter with the regularisation parameter C

C_values = [.1 , 1 , 5 , 10 ]
for gamma in gamma_values :
	for c in C_values :
		clf_svm =  SVC( kernel = 'rbf' , gamma = gamma , C = c )
		clf_svm.fit( x_train , y_train )
		print ("\n\n")
		print ("for c = %.2f and gamma = %4f  score is = %f on training set "%( c , gamma , clf_svm.score(x_train , y_train)))
		print ("for c = %.2f and gamma = %4f  score is = %f on test set " %(c , gamma ,clf_svm.score(x_test , y_test)))

## checking the value of SVM when kernel = polynomial

for gamma in gamma_values :
	clf_svm = SVC( kernel = 'poly' , gamma = gamma )
	clf_svm.fit(x_train , y_train)
	print ("\n\n")
	print ("for gamma = %.4f score on training set is %.5f " % (gamma , clf_svm.score( x_train , y_train )))
	print ("for gamma = %.4f score on test set is %.5f " % (gamma , clf_svm.score(x_test , y_test)))

from sklearn.datasets import load_breast_cancer 

load_data = load_breast_cancer()
x_cancer  , y_caner =  load_breast_cancer( return_X_y = True)

gamma_values = [.001 , 1 , 5 , 10 ]
for gamma in gamma_values:
	clf_svm =  SVC(gamma = gamma , kernel = "rbf")
	clf_svm.fit(x_train , y_train)
	print ("for gamma = %.4f score on training set is %.5f " % (gamma , clf_svm.score( x_train , y_train )))
	print ("for gamma = %.4f score on test set is %.5f " % (gamma , clf_svm.score( x_test , y_test )))

## using the normalization for drastic change in the score 

from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()
x_scaled_train = scaler.fit_transform(x_train)
x_scaled_test  = scaler.transform(x_test)

for gamma in gamma_values:

	for k in ["rbf" , "poly"]:
		clf_svm = SVC( gamma = gamma , kernel = k)
		clf_svm.fit(x_scaled_train , y_train)
		print ("\n\n")
		print ("for gamma = %.4f score on training for %s  set is %.5f " % (gamma , k , clf_svm.score( x_scaled_train , y_train )))
		print ("for gamma = %.4f score on test for %s set is %.5f " % (gamma , k , clf_svm.score( x_scaled_test , y_test )))

