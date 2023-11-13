import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        median = np.median(x, axis=0)
        mad = np.mean(np.abs(x - median), axis=0)
        return mad
        ####


    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc = np.median(features, axis=0) # YOUR CODE HERE
        self.scale = (np.sum(np.abs(features - self.loc), axis=0)) / len(features)
        # YOUR CODE HERE
        ####


    def logpdf(self, value):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        return -np.log(2 *self.scale) - np.abs(value - self.loc) / self.scale


    ####
        
    
    def pdf(self, value):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(value))
