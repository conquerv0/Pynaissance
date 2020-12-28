import numpy as np
from scipy.stats import multivariate_normal

class GDA:
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target labels
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.phi = np.zeros((n_classes, 1))
        self.means = np.zeros((n_classes, n_features))
        self.sigma = 0
        for i in range(n_classes):
            indexes = np.flatnonzero(y == self.classes[i])

            self.phi[i] = len(indexes) / n_samples
            self.means[i] = np.mean(X[indexes], axis=0)
            self.sigma += np.cov(X[indexes].T) * (len(indexes) - 1)

        self.sigma /= n_samples

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample.
        """
        pdf = lambda mean: multivariate_normal.pdf(X, mean=mean, cov=self.sigma)
        y_probs = np.apply_along_axis(pdf, 1, self.means) * self.phi

        return self.classes[np.argmax(y_probs, axis=0)]
