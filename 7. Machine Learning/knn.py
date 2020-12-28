  
import numpy as np

class KNN:
    def fit(self, X, y, n_neighbors, distance):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples, n_classes)
            Target values
        n_neighbors : Number of neighbors
        distance : Distance algorithm, see also distance.py
        """
        self.X = X
        self.y = y
        self.n_neighbors = n_neighbors
        self.distance = distance

    def __predict(self, x):
        distances = self.distance(x, self.X)
        nearest_items = np.argpartition(distances, self.n_neighbors - 1)[:self.n_neighbors]
        return np.argmax(np.bincount(self.y[nearest_items].astype(int)))

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
        return np.apply_along_axis(self.__predict, 1, X)
