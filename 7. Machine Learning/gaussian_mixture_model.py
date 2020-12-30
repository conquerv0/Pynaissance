import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def fit(self, X, n_clusters, epochs):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_clusters : The number of clusters
        epochs : The number of epochs
        Returns
        -------
        y : shape (n_samples, 1)
            Predicted cluster label per sample.
        """
        n_samples, n_features = X.shape
        self.n_clusters = n_clusters
        self.phi = np.full(self.n_clusters, 1 / self.n_clusters)
        self.means = X[np.random.choice(n_samples, self.n_clusters)]
        self.sigma = np.repeat(np.expand_dims(np.cov(X.T), axis=0), 3, axis=0)

        for _ in range(epochs):
            y_probs = self.score(X)

            n_classes = np.sum(y_probs, axis=0)
            self.phi = n_classes / n_samples

            for i in range(self.n_clusters):
                self.means[i] = np.sum(y_probs[:, i].reshape((-1, 1)) * X, axis=0) / n_classes[i]

                diff1 = (X - self.means[i])[:, :, np.newaxis]
                diff2 = np.transpose(diff1, axes=(0, 2, 1)) * y_probs[:, i].reshape(-1, 1, 1)
                self.sigma[i] = np.tensordot(diff1, diff2, axes=(0, 0)).reshape((n_features, n_features)) / n_classes[
                    i]

                '''
                for j in range(n_samples):
                    diff = (X[j] - self.__means[i]).reshape(-1, 1)
                    self.sigma[i] += y_probs[j, i] * diff.dot(diff.T)
                self.sigma[i] /= n_classes[:, i]
                '''

        return self.predict(X)

    def score(self, X):
        n_samples = X.shape[0]

        X_probs = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            X_probs[:, i] = multivariate_normal.pdf(X, mean=self.means[i], cov=self.sigma[i])

        return self.phi * X_probs / np.sum(self.phi * X_probs, axis=1, keepdims=True)

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples, 1)
            Predicted cluster label per sample.
        """
        return np.argmax(self.score(X), axis=1)