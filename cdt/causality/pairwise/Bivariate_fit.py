"""
Bivariate fit model
Author : Olivier Goudet
Date : 7/06/17
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import numpy as np
from .model import PairwiseModel
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
from sklearn.linear_model import LinearRegression


class BivariateFit(PairwiseModel):
    """
    Bivariate Fit model.
    Based itself on a best-fit criterion based on a Gaussian Process regressor.
    Used as weak baseline.
    """
    def __init__(self, ffactor=2, maxdev=3, minc=12):
        super(BivariateFit, self).__init__()

    def predict_proba(self, a, b, **kwargs):
        """ Infer causal relationships between 2 variables using regression.

        Args:
            a (numpy.ndarray): Variable 1
            b (numpy.ndarray): Variable 2

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        return self.b_fit_score(b, a) - self.b_fit_score(a, b)

    def b_fit_score(self, x, y):
        """ Computes the cds statistic from variable 1 to variable 2

        Args:
            a (numpy.ndarray): Variable 1
            b (numpy.ndarray): Variable 2

        Returns:
            float: BF fit score
        """
        x = np.reshape(scale(x), (-1, 1))
        y = np.reshape(scale(y), (-1, 1))
        gp = GaussianProcessRegressor().fit(x, y)
        y_predict = gp.predict(x)
        error = mean_squared_error(y_predict, y)

        return error
