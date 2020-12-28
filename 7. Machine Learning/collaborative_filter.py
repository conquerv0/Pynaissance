import numpy as np
import matplotlib.pyplot as plt

class CollaborativeFiltering:
    def fit(self, X, y, dimension, learning_rate, epochs):
        """
        Parameters
        ----------
        X : shape (n_samples, 2)
            Training data, column 1 is user id, column 2 is item id
        y : shape (n_samples,)
            Rating
        learning_rate : learning rate
        epochs : The number of epochs
        """
        n_samples = X.shape[0]
        user_id = X[:, 0]
        item_id = X[:, 1]

        self.user_items = np.unique(user_id)
        self.item_items = np.unique(item_id)

        n_users = len(self.user_items)
        n_items = len(self.item_items)

        self.user_vector = np.random.uniform(size=(n_users, dimension))
        self.user_bias = np.zeros((n_users, 1))
        self.item_vector = np.random.uniform(size=(n_items, dimension))
        self.item_bias = np.zeros((n_items, 1))

        loss = []
        for _ in range(epochs):
            index = np.random.randint(0, n_samples)

            user_index = np.flatnonzero(self.user_items == user_id[index])
            item_index = np.flatnonzero(self.item_items == item_id[index])

            r = (self.user_vector[user_index].dot(self.item_vector[item_index].T) + self.user_bias[user_index] + self.item_bias[item_index] - y[index])

            loss.append(r.ravel() ** 2)

            user_vector_new = self.user_vector[user_index] - learning_rate * r * self.item_vector[item_index]
            self.user_bias[user_index] -= learning_rate * r
            item_vector_new = self.item_vector[item_index] - learning_rate * r * self.user_vector[user_index]
            self.item_bias[item_index] -= learning_rate * r
            
            self.user_vector[user_index] = user_vector_new
            self.item_vector[item_index] = item_vector_new
            
        plt.plot(loss)
        plt.show()

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, 2)
            Predicting data, column 1 is user id, column 2 is item id
        Returns
        -------
        y : shape (n_samples,)
            Predicted rating per sample.
        """
        n_samples = X.shape[0]
        user_id = X[:, 0]
        item_id = X[:, 1]

        y = np.zeros(n_samples)
        for i in range(n_samples):
            user_index = np.flatnonzero(self.user_items == user_id[i])
            item_index = np.flatnonzero(self.item_items == item_id[i])
            y[i] = self.user_vector[user_index].dot(self.item_vector[item_index].T) + self.user_bias[user_index] + self.item_bias[item_index]

        return y
