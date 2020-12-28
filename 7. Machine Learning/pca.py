  
import numpy as np
from tools import metrics

class PCA:
    def __init__(self, n_components, whiten=False, method='', visualize=False):
        """
        Parameters
        ----------
        n_components : Number of components to keep
        whiten : Whitening
        method : whether SVD us applied or not
        visualize : Plot scatter if n_components equals 2
        """
        self.n_components = n_components
        self.whiten = whiten
        self.method = method
        self.visualize = visualize

    def fit_transform(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        Returns
        -------
        X : shape (n_samples, n_components)
            The projected data with dimensionality reduction
        """
        n_samples = X.shape[0]

        self.mean = np.mean(X, axis=0)
        X_sub_mean = X - self.mean

        if self.method == 'svd':
            u, s, vh = np.linalg.svd(X_sub_mean)
            self.__eig_values = (s ** 2)[:self.n_components]
            self.__eig_vectors = vh.T[:, :self.n_components]
        else:
            conv = X_sub_mean.T.dot(X_sub_mean)
            eig_values, eig_vectors = np.linalg.eigh(conv)
            self.eig_values = eig_values[::-1][:self.n_components]
            self.eig_vectors = eig_vectors[:, ::-1][:, :self.n_components]
        
        if self.whiten:
            self.std = np.sqrt(self.__eig_values.reshape((1, -1)) / (n_samples - 1))

        return self.transform(X)
        
    def transform(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        """
        X_sub_mean = X - self.mean

        pc = X_sub_mean.dot(self.__eig_vectors)

        if self.whiten:
            pc /= self.std

        if self.n_components == 2 and self.visualize:
            metrics.scatter_feature(pc)

        return pc

      
class KernelPCA:
    def __init__(self, n_components, kernel_func, sigma=1, visualize=False):
        """
        Parameters
        ----------
        n_components : Number of components to keep
        kernel_func : kernel algorithm see also kernel.py
        sigma : Parameter for rbf kernel
        visualize : Plot scatter if n_components equals 2
        """
        self.n_components = n_components
        self.kernel_func = kernel_func
        self.sigma = sigma
        self.visualize = visualize

    def fit_transform(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        """
        self.X = X
        n_samples = self.X.shape[0]
        
        K = self.kernel_func(self.X, self.X, self.sigma)
        self.K_row_mean = np.mean(K, axis=0)
        self.K_mean = np.mean(self.K_row_mean)

        I = np.full((n_samples, n_samples), 1 / n_samples)
        K_hat = K - I.dot(K) - K.dot(I) + I.dot(K).dot(I)

        eig_values, eig_vectors = np.linalg.eigh(K_hat)
        self.eig_values = eig_values[::-1][:self.n_components]
        self.eig_vectors = eig_vectors[:, ::-1][:, :self.n_components]

        #pc = self.eig_vectors * np.sqrt(self.eig_values)

        return self.transform(X)
    
    def kernel_centeralization(self, kernel):
        kernel -= self.K_row_mean
        K_pred_cols = (np.sum(kernel, axis=1) / self.K_row_mean.shape[0]).reshape((-1, 1))
        kernel -= K_pred_cols
        kernel += self.K_mean

        return kernel

    def transform(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        """
        kernel = self.kernel_func(self.X, X, self.sigma)
        kernel = self.kernel_centeralization(kernel)
        pc = kernel.dot(self.eig_vectors / np.sqrt(self.eig_values))

        if self.n_components == 2 and self.visualize:
            metrics.scatter_feature(pc)

        return pc


class ZCAWhiten:
    def __init__(self, method=''):
        """
        Parameters
        ----------
        method : SVD or not
        """
        self.method = method
    
    def fit_transform(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        Returns
        -------
        X : shape (n_samples, n_components)
            The data whitened
        """
        n_samples = X.shape[0]

        self.mean = np.mean(X, axis=0)
        X_sub_mean = X - self.mean

        if self.method == 'svd':
            u, s, vh = np.linalg.svd(X_sub_mean)
            self.__eig_values = s ** 2
            self.__eig_vectors = vh.T
        else:
            conv = X_sub_mean.T.dot(X_sub_mean)
            eig_values, eig_vectors = np.linalg.eigh(conv)
            self.eig_values = eig_values[::-1]
            self.eig_vectors = eig_vectors[:, ::-1]

        self.std = np.sqrt(self.eig_values.reshape((1, -1)) / (n_samples - 1))

        return self.transform(X)

    def transform(self, X):
        """
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        X : shape (n_samples, n_components)
            The data whitened
        """
        X_sub_mean = X - self.mean

        pc = X_sub_mean.dot(self.eig_vectors)
        pc /= self.std

        return pc.dot(self.eig_vectors.T)
