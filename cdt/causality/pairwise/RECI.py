"""
Bivariate fit model
Author : Olivier Goudet
Date : 7/06/17
"""
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
import numpy as np
from .model import PairwiseModel
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
from sklearn.linear_model import LinearRegression


class RECI(PairwiseModel):
    """
    RECI.
    best-fit mse with monome regressor and [0,1] rescaling
    """
    def __init__(self, ffactor=2, maxdev=3, minc=12):
        super(RECI, self).__init__()

    def predict_proba(self, a, b, **kwargs):
        """ Infer causal relationships between 2 variables x_te and y_te using the CDS statistic

        :param a: Input variable 1
        :param b: Input variable 2
        :return: (Value : 1 if a->b and -1 if b->a)
        :rtype: float
        """
        return self.b_fit_score(b, a) - self.b_fit_score(a, b)

    def b_fit_score(self, x, y):

        x = np.reshape(minmax_scale(x), (-1, 1))
        y = np.reshape(minmax_scale(y), (-1, 1))


        poly = PolynomialFeatures(degree=3)
        poly_x = poly.fit_transform(x)

        poly_x[:,1] = 0
        poly_x[:,2] = 0

        regressor = LinearRegression()
        regressor.fit(poly_x, y)

        y_predict = regressor.predict(poly_x)

        
        error = mean_squared_error(y_predict, y)

        return error
