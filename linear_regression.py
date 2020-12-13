# Linear Regression

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import sys

df = pd.read_csv(r"student-mat.csv", sep=";")
data = df[["G1", "G2", "G3", "studytime", "failures", "absences"]]    # Needed data

predict = "G3"

X = np.array(data.drop([predict], 1))    # Features
y = np.array(data[predict])    # Label array

X_train, y_train, X_test, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

