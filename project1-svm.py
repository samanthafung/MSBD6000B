import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math

with open('traindata1.csv', 'r') as f1: train_dataframe = pd.read_csv(f1)

with open('trainlabel1.csv','r') as f2: train_lableframe = pd.read_csv(f2)

#print(train_dataframe.head())
#print(train_lableframe.head())

X = train_dataframe
y = train_lableframe

#Normalization
#Subtract the mean for each feature
X -= np.mean(X, axis=0)
#Divide each feature by its Standard deviation
X /= np.std(X, axis=0)

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Train Model
#regression_model = LinearRegression()
regression_model = svm.SVC(kernel='linear', C=1, gamma=1)
regression_model.fit(X_train, y_train)


print("The Modeling score is %f" %regression_model.score(X_test, y_test))

y_predict = regression_model.predict(X_test)

regression_model_mse = mean_squared_error(y_predict, y_test)

print ("The mean squared error is %f" %regression_model_mse)

with open('testdata1.csv', 'r') as f3: test_dataframe = pd.read_csv(f3)

ans = regression_model.predict(test_dataframe)

np.savetxt("project1_20057566.csv", ans, delimiter=",")





