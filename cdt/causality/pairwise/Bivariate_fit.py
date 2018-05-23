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


class BivariateFit(PairwiseModel):
    """
    Bivariate Fit model.
    Based itself on a best-fit criterion based on a regressor.
    """
    def __init__(self, ffactor=2, maxdev=3, minc=12):
        super(BivariateFit, self).__init__()

    def predict_proba(self, a, b, **kwargs):
        """ Infer causal relationships between 2 variables x_te and y_te using the CDS statistic

        :param a: Input variable 1
        :param b: Input variable 2
        :return: (Value : 1 if a->b and -1 if b->a)
        :rtype: float
        """
        return self.b_fit_score(b, a) - self.b_fit_score(a, b)

    def b_fit_score(self, x, y):
        """ Computes the cds statistic from variable 1 to variable 2

        :param x: Input, seen as cause
        :param y: Input, seen as effect
        :return: CDS statistic between x_te and y_te
        """
        x = np.reshape(scale(x), (-1, 1))
        y = np.reshape(scale(y), (-1, 1))

        gp = GaussianProcessRegressor().fit(x, y)
        y_predict = gp.predict(x)
        error = mean_squared_error(y_predict, y)

        return error
