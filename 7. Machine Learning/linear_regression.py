import numpy as np
import matplotlib.pyplot as plt
from tools import optimizer, regularizer


class LinearRegressionGradientDescent:
    def __init__(self, X):
        self.weights = np.zeros(X.shape)
        self.bias = 0

    def fit(self, X, y, epochs, optimizer, regularizer):
        """
        @param X: Training data, with shape(n_samples, n_features)
        @param y: Target values, with shape(n_samples, 1)
        @param epochs: The number of epochs.
        @param optimizer: the optimzer chosen.
        @param regularizer: the regularizer chosen.
        """
        n_samples, n_features = X.shape
        regularizer = regularizer.Regularizer(0)

        for _ in range(epochs):
            h = self.predict(X)

            dW = X.T.dot(h - y) / n_samples + regularizer.regularize(self.weights)
            db = np.mean(h - y)
            dW, db = optimizer.optimize([dW])


    def predict(self, X):
        """
        @param X: Data with shape (n_samples, n_features)
        @param y: Prediction with shape (n_samples, 1)
        """
        return X.dot(self.weights) + self.bias
