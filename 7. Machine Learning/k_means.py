import numpy as np
from tools import distance


class KMeans:
    @property
    def centers(self):
        return self.centers

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
        y : shape (n_samples,)
            Predicted cluster label per sample.
        """
        n_samples = X.shape[0]
        self.centers = X[np.random.choice(n_samples, n_clusters)]

        for _ in range(epochs):
            labels = self.predict(X)
            self.centers = [np.mean(X[np.flatnonzero(labels == i)], axis=0) for i in range(n_clusters)]

        return self.predict(X)

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        """
        distances = np.apply_along_axis(distance.euclidean_distance, 1, self.centers, X).T
        return np.argmin(distances, axis=1)

    @centers.setter
    def centers(self, value):
        self._centers = value


class BisectingKMeans(KMeans):
    def fit(self, X, n_clusters):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_clusters : The number of clusters
        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        """
        n_samples = X.shape[0]

        data = X
        clusters = []
        while True:
            model = KMeans()
            label = model.fit(data, 2, 100)

            clusters.append(np.flatnonzero(label == 0))
            clusters.append(np.flatnonzero(label == 1))

            if len(clusters) == n_clusters:
                break

            sse = [np.var(data[cluster]) for cluster in clusters]
            data = data[clusters[np.argmax(sse)]]
            del clusters[np.argmax(sse)]

        y = np.zeros(n_samples)
        for i in range(len(clusters)):
            y[clusters[i]] = i

        return y