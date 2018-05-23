"""Methods using standard feature selection algorithms to recover the undirected graph.

Using the sklearn tools
Author: Olivier Goudet
"""

from .model import FeatureSelectionModel
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from skrebate import ReliefF
import numpy as np
from sklearn.linear_model import ARDRegression


class RFECV_linearSVR(FeatureSelectionModel):
    """ RandomizedLasso from scikit-learn
    """

    def __init__(self):
        super(RFECV_linearSVR, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        estimator = SVR(kernel='linear')
        selector = RFECV(estimator, step=1)
        selector = selector.fit(df_features.as_matrix(), df_target.as_matrix()[:, 0])

        return selector.grid_scores_


class LinearSVR_L2(FeatureSelectionModel):
    """ RandomizedLasso from scikit-learn
    """

    def __init__(self):
        super(LinearSVR_L2, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        C = kwargs.get("C", 0.1)
        lsvc = LinearSVR(C=C).fit(df_features.as_matrix(), df_target.as_matrix())

        return np.abs(lsvc.coef_)


class DecisionTree_regressor(FeatureSelectionModel):
    """ RandomizedLasso from scikit-learn
    """

    def __init__(self):
        super(DecisionTree_regressor, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        X = df_features.as_matrix()
        y = df_target.as_matrix()
        regressor = DecisionTreeRegressor()
        regressor.fit(X, y)

        return regressor.feature_importances_


class ARD_Regression(FeatureSelectionModel):
    def __init__(self):
        super(ARD_Regression, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        X = df_features.as_matrix()
        y = df_target.as_matrix()
        clf = ARDRegression(compute_score=True)
        clf.fit(X, y.ravel())

        return np.abs(clf.coef_)


class RRelief(FeatureSelectionModel):
    def __init__(self):
        super(RRelief, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        X = df_features.as_matrix()
        y = df_target.as_matrix()[:, 0]
        rr = ReliefF()
        rr.fit(X, y)

        return rr.feature_importances_
