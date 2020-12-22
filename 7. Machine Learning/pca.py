  
import numpy as np
import metrics

class PCA:
    def __init__(self, n_components, whiten=False, method='', visualize=False):
        '''
        Parameters
        ----------
        n_components : Number of components to keep
        whiten : Whitening
        method : whether SVD us applied or not
        visualize : Plot scatter if n_components equals 2
        '''
        self.__n_components = n_components
        self.__whiten = whiten
        self.__method = method
        self.__visualize = visualize

    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        Returns
        -------
        X : shape (n_samples, n_components)
            The projected data with dimensionality reduction
        '''
        n_samples = X.shape[0]

        self.__mean = np.mean(X, axis=0)
        X_sub_mean = X - self.__mean

        if self.__method == 'svd':
            u, s, vh = np.linalg.svd(X_sub_mean)
            self.__eig_values = (s ** 2)[:self.__n_components]
            self.__eig_vectors = vh.T[:, :self.__n_components]
        else:
            conv = X_sub_mean.T.dot(X_sub_mean)
            eig_values, eig_vectors = np.linalg.eigh(conv)
            self.__eig_values = eig_values[::-1][:self.__n_components]
            self.__eig_vectors = eig_vectors[:, ::-1][:, :self.__n_components]
        
        if self.__whiten:
            self.__std = np.sqrt(self.__eig_values.reshape((1, -1)) / (n_samples - 1))

        return self.transform(X)
        
    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        X_sub_mean = X - self.__mean

        pc = X_sub_mean.dot(self.__eig_vectors)

        if self.__whiten:
            pc /= self.__std

        if self.__n_components == 2 and self.__visualize:
            metrics.scatter_feature(pc)

        return pc
