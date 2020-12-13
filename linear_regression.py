# Linear Regression

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


df = pd.read_csv(r"student-mat.csv", sep=";")
data = df[["G1", "G2", "G3", "studytime", "failures", "absences"]]    # Needed data

predict = "G3"

X = np.array(data.drop([predict], 1))    # Features
y = np.array(data[predict])    # Label array

# Separating data sets for training and testing
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Creating and training the model
# linear = linear_model.LinearRegression()    # Loads LR model
# linear.fit(X_train, y_train)    # Finds the best fit line
#
# accuracy = linear.score(X_test, y_test)

# print(accuracy)
# print(linear.coef_)    # Gradient for each feature- 5-dimensional
# print(linear.intercept_)
pickle_in = open(r"studentmodel.pickle", "rb")    # Open stored pickle
linear = pickle.load(pickle_in)

print(pickle_in)
print(linear)

predictions = linear.predict(X_test)    # Make predictions

for x in range(len(predictions)):
    print(f"Predicted Grade:{predictions[x]}, Test Data: {X_test[x]}, Actual Grade: {y_test[x]}")

# Store model
# with open(r"studentmodel.pickle", "wb") as f:
#     pickle.dump(linear, f)
