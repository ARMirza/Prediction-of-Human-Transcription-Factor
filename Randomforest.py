from sklearn.datasets import load_iris
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("finalData.csv")

X = df.iloc[:,1:28] #features
Y = df['Factor'] #features


X,Y = shuffle(X,Y,random_state = 10)
#print(Y)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.4,random_state = 40)
#print(x_train.shape)

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=10,
                               bootstrap = 'True',
                               max_features = 'auto').fit(x_train, y_train) #"auto", "sqrt" or "log2".

print('Accuracy of Random Forest Classifier on training set: {:.2f}'
     .format(model.score(x_train, y_train)))
print('Accuracy of Random Forest Classifier on test set: {:.2f}'
     .format(model.score(x_test, y_test)))






    


