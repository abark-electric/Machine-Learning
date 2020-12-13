# Linear Regression (supervised)

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


class Regression:
    def __init__(self, filename, feature_array, label):
        self.data = pd.read_csv(filename)[feature_array]
        self.predict = label
        self.X = np.array(self.data.drop([self.predict], 1))    # Features
        self.y = np.array(self.data[self.predict])  # Label array

    def separate_dataset(self):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.X, self.y, test_size=0.1)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def create_and_train_model(X_train, X_test, y_train, y_test):
        linear = linear_model.LinearRegression()  # Loads LR model
        linear.fit(X_train, y_train)  # Finds the best fit line
        accuracy = linear.score(X_test, y_test)
        return accuracy, linear

    @staticmethod
    def export_model(pickle_filename, model):
        with open(pickle_filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(pickle_filename):
        pickle_in = open(pickle_filename, "rb")  # Open stored pickle
        model = pickle.load(pickle_in)
        return model

    @staticmethod
    def make_predictions(test_data, model, X_test, y_test):
        predictions = model.predict(test_data)  # Make predictions

        for x in range(len(predictions)):
            print(f"Predicted Grade:{predictions[x]}, Test Data: {X_test[x]}, Actual Grade: {y_test[x]}")


    def plot_data(self, feature, label):
        p = "G1"
        style.use("ggplot")
        plt.scatter(self.data[feature], self.data[label])
        plt.xlabel(feature)
        plt.ylabel(label)
        plt.show()
