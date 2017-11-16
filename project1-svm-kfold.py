import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
import math

with open('traindata1.csv', 'r') as f1: train_dataframe = pd.read_csv(f1)

with open('trainlabel1.csv','r') as f2: train_lableframe = pd.read_csv(f2)

#print(train_dataframe.head())
#print(train_lableframe.head())

X = train_dataframe.as_matrix()
y = train_lableframe.as_matrix()

#Normalization
#Subtract the mean for each feature
X -= np.mean(X, axis=0)
#Divide each feature by its Standard deviation
X /= np.std(X, axis=0)

# Split X and y into X_
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# K-fold
kf =KFold(n_splits=10, random_state=3, shuffle=True)
kf.get_n_splits(X)
print(kf)

for train_index, test_index in kf.split(X,y):
     print("Train:", train_index, "Test:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = svm.SVC(kernel='rbf', gamma=0.01, random_state=3)
     model.fit(X_train, y_train)

# Perform 5-fold cross validation
# scores = cross_val_score(model, X, y, cv=5)
# print("Cross-validated scores:", scores)

print("The Modeling score is %f" %model.score(X_test, y_test))

predictions = model.predict(X_test)

# predictions = cross_val_predict(model, X, y, cv=5)
##y_predict = regression_model.predict(X_test)

model_mse = mean_squared_error(predictions, y_test)

print ("The mean squared error is %f" %model_mse)
# accuracy = metrics.r2_score(y, predictions)
# print("Cross-Predicted Accuracy:", accuracy)

with open('testdata1.csv', 'r') as f3: test_dataframe = pd.read_csv(f3)

ans = model.predict(test_dataframe)

np.savetxt("project1_20057566.csv", ans, delimiter=",")





