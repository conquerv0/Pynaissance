import numpy as np

class DecisionStump():
    def update_parameter(self, h, feature_index, threshold, direction, err_val):
      self.min_err_val = err_val
      self.feature_index = feature_index
      self.threshold = threshold
      self.direction = direction
      
  
    def select_direction(self, feature_index, threshold, X, y, w):
        for direction in ['greater', 'less']:
            h = np.ones_like(y)

            if direction == 'greater':
                h[np.flatnonzero(X[:, feature_index] < threshold)] = -1
            else:
                h[np.flatnonzero(X[:, feature_index] >= threshold)] = -1

            err_value = np.sum(w[np.flatnonzero(h != y)])
            if err_value < self.min_err_val:
                self.update_parameter(h, feature_index, threshold, 'greater', err_value)
          
    def select_threshold(self, feature_index, X, y, w):
        n_samples = X.shape[0]
       
        for i in range(n_samples):
            self.select_direction(feature_index, X[i, feature_index], X, y, w)
            
    def select_features(self, X, y, w):
        n_features = X.shape[1]
        for i in range(n_features):
            self.select_threshold(i, X, y, w)
            
    def fit(self, X, y, w):
        self.feature_index = None
        self.threshold = None
        self.direction = None
        self.min_err_val = np.inf
        
        self.select_features(X, y, w)
