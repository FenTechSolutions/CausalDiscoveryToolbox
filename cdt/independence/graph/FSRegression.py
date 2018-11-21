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
from sklearn.linear_model import ARDRegression as ard


class RFECVLinearSVR(FeatureSelectionModel):
    """ Recursive Feature elimination with cross validation,
    with support vector regressors

    .. note::
       Ref: Guyon, I., Weston, J., Barnhill, S., & Vapnik, V.,
       “Gene selection for cancer classification using support vector machines”,
       Mach. Learn., 46(1-3), 389–422, 2002.
    """

    def __init__(self):
        super(RFECVLinearSVR, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms

        Returns:
            list: scores of each feature relatively to the target
        """
        estimator = SVR(kernel='linear')
        selector = RFECV(estimator, step=1)
        selector = selector.fit(df_features.values, df_target.values[:, 0])

        return selector.grid_scores_


class LinearSVRL2(FeatureSelectionModel):
    """ Feature selection with Linear Support Vector Regression."""

    def __init__(self):
        super(LinearSVRL2, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, C=.1, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms
            C (float): Penalty parameter of the error term

        Returns:
            list: scores of each feature relatively to the target
        """
        lsvc = LinearSVR(C=C).fit(df_features.values, df_target.values)

        return np.abs(lsvc.coef_)


class DecisionTreeRegression(FeatureSelectionModel):
    """ Feature selection with decision tree regression."""

    def __init__(self):
        super(DecisionTreeRegression, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms

        Returns:
            list: scores of each feature relatively to the target
        """
        X = df_features.values
        y = df_target.values
        regressor = DecisionTreeRegressor()
        regressor.fit(X, y)

        return regressor.feature_importances_


class ARD(FeatureSelectionModel):
    """ Feature selection with Bayesian ARD regression."""
    def __init__(self):
        super(ARD, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms

        Returns:
            list: scores of each feature relatively to the target
        """
        X = df_features.values
        y = df_target.values
        clf = ard(compute_score=True)
        clf.fit(X, y.ravel())

        return np.abs(clf.coef_)


class RRelief(FeatureSelectionModel):
    """ Feature selection with RRelief."""
    def __init__(self):
        super(RRelief, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms

        Returns:
            list: scores of each feature relatively to the target
        """
        X = df_features.values
        y = df_target.values[:, 0]
        rr = ReliefF()
        rr.fit(X, y)

        return rr.feature_importances_
