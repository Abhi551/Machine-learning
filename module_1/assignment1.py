import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

## loading data in load_data
load_data =  load_breast_cancer()

#print (load_data)

print (load_data.keys())

## QUESTION 0
## how many features does load_data have and what is their names??
def features():
	for i in load_data.feature_names:
		print (i)
	return (len(load_data.feature_names))

no_of_features = features()
print ("number of features are " , no_of_features)

#print (load_data['target_names'])

#QUESTION 1

## it is always good to convert the data into DataFrame for better handling
## converting the sklearn data in pandas dataframe 

def answer_1():
	#print (load_data.data[:2])
	df_cancer = pd.DataFrame(load_data['data'] , columns = load_data['feature_names'] )
	#print (df_cancer.head())
	print (df_cancer.shape)
	return (df_cancer)


df_cancer = answer_1()

## including the target in df_cancer 

df_cancer['target'] = load_data['target']

print (df_cancer.shape)
## QUESTION 2

## what is class distribution
## return a instance of malignant (target = 0) and benign (target = 1)

def answer_2():
	pass


print (df_cancer.head())

df_cancer.ix[:,df_cancer.columns!='target']
print (df_cancer.ix)

def answer_3():

	## to use train_test_split we have to pass X and Y targets since our goal is to tell the type of cancer 
	X = df_cancer[:30]
	print (X.shape)
	Y = df_cancer[30]
	print (Y.shape)




def answer_4():
	X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = .75 , train_size = .25)





