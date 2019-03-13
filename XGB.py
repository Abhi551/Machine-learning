from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

## dataset
dataset = loadtxt("dataset.csv", delimiter = ",")

## train_test_split
X = dataset[:,0:8]
Y = dataset[:,8]

X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size = .33 , random_state = 7)

## train the model 

model = XGBClassifier()
model.fit(X_train , y_train)

print ("check\n")
print (model)

## predict 
y_pred =  model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test , predictions)
print("\n\n accuracy =",(accuracy*100.0))