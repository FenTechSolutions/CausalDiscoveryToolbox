"""Methods using standard feature selection algorithms to recover the undirected graph.

Using the sklearn tools
Author: Olivier Goudet

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
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
       
   Example:
       >>> from cdt.independence.graph import RFECVLinearSVR
       >>> from sklearn.datasets import load_boston
       >>> boston = load_boston()
       >>> df_features = pd.DataFrame(boston['data'])
       >>> df_target = pd.DataFrame(boston['target'])
       >>> obj = RFECVLinearSVR()
       >>> output = obj.predict_features(df_features, df_target)
       >>> ugraph = obj.predict(df_features)  # Predict skeleton
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
        selector = selector.fit(df_features.values, np.ravel(df_target.values))

        return selector.grid_scores_


class LinearSVRL2(FeatureSelectionModel):
    """ Feature selection with Linear Support Vector Regression.

        Example:
            >>> from cdt.independence.graph import LinearSVRL2
            >>> from sklearn.datasets import load_boston
            >>> boston = load_boston()
            >>> df_features = pd.DataFrame(boston['data'])
            >>> df_target = pd.DataFrame(boston['target'])
            >>> obj = LinearSVRL2()
            >>> output = obj.predict_features(df_features, df_target)
            >>> ugraph = obj.predict(df_features)  # Predict skeleton
    """

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
        lsvc = LinearSVR(C=C).fit(df_features.values, np.ravel(df_target.values))

        return np.abs(lsvc.coef_)


class DecisionTreeRegression(FeatureSelectionModel):
    """ Feature selection with decision tree regression.

        Example:
            >>> from cdt.independence.graph import DecisionTreeRegression
            >>> from sklearn.datasets import load_boston
            >>> boston = load_boston()
            >>> df_features = pd.DataFrame(boston['data'])
            >>> df_target = pd.DataFrame(boston['target'])
            >>> obj = DecisionTreeRegression()
            >>> output = obj.predict_features(df_features, df_target)
            >>> ugraph = obj.predict(df_features)  # Predict skeleton

    """

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
    """ Feature selection with Bayesian ARD regression.

        Example:
            >>> from cdt.independence.graph import ARD
            >>> from sklearn.datasets import load_boston
            >>> boston = load_boston()
            >>> df_features = pd.DataFrame(boston['data'])
            >>> df_target = pd.DataFrame(boston['target'])
            >>> obj = ARD()
            >>> output = obj.predict_features(df_features, df_target)
            >>> ugraph = obj.predict(df_features)  # Predict skeleton
    """
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
    """ Feature selection with RRelief.

        Example:
            >>> from cdt.independence.graph import RRelief
            >>> from sklearn.datasets import load_boston
            >>> boston = load_boston()
            >>> df_features = pd.DataFrame(boston['data'])
            >>> df_target = pd.DataFrame(boston['target'])
            >>> obj = RRelief()
            >>> output = obj.predict_features(df_features, df_target)
            >>> ugraph = obj.predict(df_features)  # Predict skeleton
    """
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
        y = np.ravel(df_target.values)
        rr = ReliefF()
        rr.fit(X, y)

        return rr.feature_importances_
