from sklearn.linear_model import RandomizedLasso
from .model import FeatureSelectionModel
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from skrebate import ReliefF
from .HSICLasso import *
import numpy as np
from sklearn.linear_model import ARDRegression


class RandomizedLasso_model(FeatureSelectionModel):
    """ RandomizedLasso from scikit-learn
    """

    def __init__(self):
        super(RandomizedLasso_model, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        alpha = kwargs.get("alpha", 'aic')
        scaling = kwargs.get("scaling", 0.5)
        sample_fraction = kwargs.get("sample_fraction", 0.75)
        n_resampling = kwargs.get("n_resampling", 200)

        randomized_lasso = RandomizedLasso(alpha=alpha, scaling=scaling, sample_fraction=sample_fraction,
                                           n_resampling=n_resampling)
        randomized_lasso.fit(df_features.as_matrix(), df_target.as_matrix())

        return randomized_lasso.scores_


class RFECV_linearSVR(FeatureSelectionModel):
    """ RandomizedLasso from scikit-learn
    """

    def __init__(self):
        super(RFECV_linearSVR, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        estimator = SVR(kernel='linear')
        selector = RFECV(estimator, step=1)

        selector = selector.fit(df_features.as_matrix(), df_target.as_matrix())

        print(selector.grid_scores_)

        return selector.grid_scores_


class linearSVR_L2(FeatureSelectionModel):
    """ RandomizedLasso from scikit-learn
    """

    def __init__(self):
        super(linearSVR_L2, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        C = kwargs.get("C", 0.1)

        lsvc = LinearSVR(C=C).fit(df_features.as_matrix(), df_target.as_matrix())

        print(np.abs(lsvc.coef_))

        return np.abs(lsvc.coef_)


class decisionTree_regressor(FeatureSelectionModel):
    """ RandomizedLasso from scikit-learn
    """

    def __init__(self):
        super(decisionTree_regressor, self).__init__()

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
        clf.fit(X, y)

        return np.abs(clf.coef_)


class RRelief(FeatureSelectionModel):
    def __init__(self):
        super(RRelief, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        X = df_features.as_matrix()
        y = df_target.as_matrix()

        rr = ReliefF()
        rr.fit(X, y)

        return rr.feature_importances_


class HSICLasso(FeatureSelectionModel):
    def __init__(self):
        super(HSICLasso, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        X = np.transpose(df_features.as_matrix())
        y = np.transpose(df_target.as_matrix())

        path, beta, A, lam = hsiclasso(X, y, numFeat=5)

        return beta
