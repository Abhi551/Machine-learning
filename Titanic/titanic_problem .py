
import sklearn
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.linear_model import LinearRegression , Ridge ,Lasso , LogisticRegression
from sklearn.model_selection import GridSearchCV

df = pd.read_csv()


df = pd.read_csv("train.csv")
new_df = df.copy()
total = len(df.Name)
for col in df.columns:
    percent_na = (float(df[col].isna().sum())/total)*100
    if percent_na>70.0:
        new_df = new_df.drop(col , axis = 1)

print (new_df.head(10))




print (new_df.shape)



print (new_df.info())


# now filling up the missing data




new_df.Embarked.value_counts()



print (new_df.Age.mean())
print (new_df.Age.std())




new_df.Embarked = new_df.Embarked.fillna('S')




print (new_df.info())




# now for age we need to have better way of looking things , since the data is much mixed up and diverse , using 
# graphical interpretation would be of help




print (new_df.Age.describe())




x = new_df.Age
x = x.dropna()
ax =  sns.distplot(x , kde = True , color = 'red')




#  why mean to be avoided here?
#  because std dev is very high as nearly half of mean , using mean will naive 



new_df.Age = new_df.Age.interpolate(method = 'linear')



print new_df.Age.describe()



print new_df.head()



ax = sns.distplot(new_df.Age)



print new_df.head()
#  passengerid , name and ticket seems to be of no use and so its better to drop these 




new_df = new_df.drop(['PassengerId' , "Name" , 'Ticket'] , axis = 1)
print new_df.head()



# sns.set(style = 'darkgrid')
sns.countplot( x = 'Pclass',data = new_df)


sns.countplot(y = 'Pclass' , hue = 'Sex' , data = new_df)



sns.set(style = "whitegrid")
g = sns.PairGrid(data = new_df , x_vars = 'Pclass' ,y_vars = 'Survived' , height = 4)
g.map(sns.pointplot)
g.set(ylim = (0,1))



g = sns.PairGrid(new_df ,  x_vars = 'Sex' , y_vars = 'Survived' ,size = 5)
g.map(sns.pointplot)
g.set(ylim = (0,1))




new_df['child'] = new_df['Age'].apply(lambda x:1 if x<15 else 0)
new_df['child'].value_counts()
g = sns.PairGrid(new_df , y_vars = 'Survived' , x_vars = 'child' , size = 5)
g.map(sns.pointplot)
g.set(ylim=(0,1))



g = sns.pairplot(new_df , y_vars = 'Survived' , x_vars = ['SibSp' , 'Parch'], height = 5)
g.map(sns.pointplot)
g.set(ylim = (0,1))





new_df['Family'] = new_df['SibSp']+new_df['Parch']
new_df = new_df.drop(['SibSp' , 'Parch'] , axis = 1)
print new_df.head(5)



new_df['is_female'] = new_df['Sex'].apply(lambda x : 0 if x == 'male' else 1)
new_df = new_df.drop('Sex' , axis = 1)



print new_df.head()



new_df['is_alone'] = new_df['Family'].apply(lambda x : 1 if x == 0 else 0 )
print new_df.head()



new_df = pd.get_dummies(new_df , prefix = ['is'])
print new_df.head()



y = new_df['Survived']
X = new_df.drop('Survived' ,axis = 1)



X_train , X_test , y_train , y_test = train_test_split(X,y , train_size = .75 , test_size = .25)




def tuning(parameters , model):
    classifier = model()
    clf = GridSearchCV(classifier , parameters , return_train_score = True)
    clf.fit(X_train , y_train)
    df_result =  pd.DataFrame(clf.cv_results_)
    return (df_result)
    



GradBoost = GradientBoostingClassifier()
GradBoost.fit(X_train , y_train)
print GradBoost.score(X_test , y_test)



SVM_clf = SVC(kernel='rbf' , C = 1)
SVM_clf.fit(X_train , y_train)
print SVM_clf.score(X_test , y_test)



param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
log_reg = tuning(param_grid , LogisticRegression)
log_reg



# thus using c = 10 gets the best score in test
log_reg = LogisticRegression(C=11)
log_reg.fit(X_train , y_train)
print log_reg.score(X_test , y_test)



knn = KNeighborsClassifier(n_neighbors = 8 , algorithm = 'ball_tree' )
knn.fit(X_train , y_train)
print knn.score(X_test , y_test)



ridge_reg = Ridge()
ridge_reg.fit(X_train , y_train)
print ridge_reg.score(X_test , y_test)



linear_reg = LinearRegression (normalize = True )
linear_reg.fit(X_train , y_train)
print linear_reg.score(X_test , y_test)



lasso_reg = Lasso(alpha = .0000001, normalize = True )
lasso_reg.fit(X_train , y_train)
print lasso_reg.score(X_test , y_test) 




