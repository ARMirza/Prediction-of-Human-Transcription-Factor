
import pandas as pd

#from sklearn.datasets import load_iris
import numpy as np 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("finalData.csv")

X = df.iloc[:,1:28] #features
Y = df['Factor'] #features

feature_scaler = StandardScaler()


X,Y = shuffle(X,Y,random_state = 40)
#print(Y)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.5,random_state = 10)
#print(x_train.shape)
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)

"""
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(acc)
"""
model = LogisticRegression(solver='liblinear').fit(x_train,y_train)
print('Accuracy of LogisticRegression on training set: {:.2f}'
     .format(model.score(x_train, y_train)))
print('Accuracy of LogisticRegression on test set: {:.2f}'
     .format(model.score(x_test, y_test)))




    

