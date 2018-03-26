from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


x_c , y_c = make_classification(n_samples = 100 , n_features =2 , n_redundant = 0 , n_informative = 2 ,
								n_clusters_per_class = 1 , flip_y = .1 , class_sep = .5 , random_state = 0)

x_train , x_test , y_train , y_test = train_test_split(x_c , y_c , test_size = .25)

C_values = [.0001 , 100]

## variation of c parameter in Linear Support Vector Classifier

for c in C_values :
	clf = LinearSVC(C = c)
	clf.fit(x_train , y_train)

	print (" the training score for c = %d is %.5f " %( c , clf.score(x_train , y_train)))
	print (" the test score for c = %d is %.5f " %( c , clf.score(x_test , y_test)))
	print ("\n\n")
## applying the Linear Support Vector Classifier for real datasets

cancer = load_breast_cancer()
(x_cancer , y_cancer ) = load_breast_cancer(return_X_y =True)

x_train , x_test , y_train , y_test = train_test_split( x_cancer , y_cancer )

clf = LinearSVC()

clf.fit( x_train , y_train )

print (" the training score is %.5f " %( clf.score(x_train , y_train)))
print (" the test score is %.5f " %( clf.score(x_test , y_test)))

