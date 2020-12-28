import numpy as np
import matplotlib.pyplot as plt
import metrics
import regularizer
import scipy

class LogisticRegressionGradientDescent:
    def __init__(self, debug=True):
        self.debug = debug

    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.Regularizer(0)):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values, 1 or 0
        epochs : The number of epochs
        optimizer : Optimize algorithm, see also optimizer.py
        regularizer : Regularize algorithm, see also regularizer.py
        """
        n_samples, n_features = X.shape

        self.W = np.zeros(n_features)
        self.b = 0

        if self.debug:
            accuracy = []
            loss = []

        for _ in range(epochs):
            h = self.score(X)

            g_W = X.T.dot(h - y) / n_samples + regularizer.regularize(self.W)
            g_b = np.mean(h - y)
            g_W, g_b = optimizer.optimize([g_W, g_b])
            self.W -= g_W
            self.b -= g_b

            if self.debug:
                h = self.score(X)
                loss.append(np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h)))
                accuracy.append(metrics.accuracy(y, np.around(h)))

        if self.debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample, 1 or 0
        """
        return np.around(self.score(X))

    def score(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples,)
            Predicted score per sample.
        """
        return scipy.special.expit(X.dot(self.W) + self.b)


class LogisticRegressionNewton:
    def __init__(self, debug=True):
        self.debug = debug

    def fit(self, X, y, epochs):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values, 1 or 0
        epochs : The number of epochs
        """
        n_features = X.shape[1]

        self.W = np.zeros(n_features)
        self.b = 0

        if self.debug:
            accuracy = []
            loss = []

        for _ in range(epochs):
            h = self.score(X)

            g_W = X.T.dot(h - y)
            A = np.diag((h * (1 - h)).ravel())
            H_W = X.T.dot(A).dot(X)
            self.W -= np.linalg.pinv(H_W).dot(g_W)
            
            g_b = np.sum(h - y)
            H_b = np.sum(h * (1 - h))
            self.b -= g_b / H_b
            
            if self.debug:
                h = self.score(X)
                loss.append(np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h)))
                accuracy.append(metrics.accuracy(y, np.around(h)))

        if self.debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample, 1 or 0
        """
        return np.around(self.score(X))

    def score(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples,)
            Predicted score per sample.
        """
        return scipy.special.expit(X.dot(self.W) + self.b)
