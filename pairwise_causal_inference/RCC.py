"""
Randomized Causation Coefficient Model
Author : David Lopez-Paz
Ref : Lopez-Paz, David and Muandet, Krikamol and Schölkopf, Bernhard and Tolstikhin, Ilya O,
     "Towards a Learning Theory of Cause-Effect Inference", ICML,2015.
"""

from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier as CLF
from sklearn.metrics import auc
import numpy as np


def rp(k, s, d):
    return np.hstack((np.vstack([si * np.random.randn(k, d) for si in s]),
                      2 * np.pi * np.random.rand(k * len(s), 1))).T


def f1(x, w):
    return np.cos(np.dot(np.hstack((x, np.ones((x.shape[0], 1)))), w))


def score(y, p):
    return (auc(y == 1, p) + auc(y == -1, -p)) / 2


class RCC(object):
    """ Randomized Causation Coefficient model """

    def __init__(self, *args, **kwargs):
        """
        Initialize the model w/ its parameters

        :param args: None
        :param kwargs: {K: number of randomized coefficients,
                        E: number of estimators,
                        L: number of min samples leaves of the estimator
                        max_depth: max depth of the model
                        n_jobs: number of jobs to be run on parallel}

        """
        np.random.seed(0)
        self.K = kwargs.pop('K', 333)
        self.E = kwargs.pop('E', 500)
        self.L = kwargs.pop('L', 20)
        self.n_jobs = kwargs.pop('n_jobs', 1)
        self.max_depth = kwargs.pop('max_depth', 1)

        self.params = {'random_state': 0, 'n_estimators': self.E, 'max_features': None,
                       'max_depth': self.max_depth, 'min_samples_leaf': self.L, 'verbose': 10, 'n_jobs': self.n_jobs}

        self.wx = rp(self.K, [0.15, 1.5, 15], 1)
        self.wy = rp(self.K, [0.15, 1.5, 15], 1)
        self.wz = rp(self.K, [0.15, 1.5, 15], 2)
        self.clf0 = None
        self.clf1 = None

    def fit(self, x_tr, y_tr):
        """ Fit the model on pairwise data

        :param x_tr: Input data
        :param y_tr: Targets
        :type x_tr: pandas.DataFrame
        :type y_tr: pandas.DataFrame
        """
        x_tr, y_tr, x_ab, y_ab = self.transform(x_tr, y_tr)
        self.fit_ftdata(x_tr, y_tr, x_ab, y_ab)

    def fit_ftdata(self, x_ft, y_ft, x_ab, y_ab):
        """ Fit the model with featurized data as input

        :param x_ft: x_featurized
        :param y_ft: y_featurized
        :param x_ab: x_inverse_featurized
        :param y_ab: y_inverse_featurized
        """
        self.clf0 = CLF(**self.params).fit(x_ft, y_ft != 0)  # causal or confounded?
        self.clf1 = CLF(**self.params).fit(x_ab, y_ab == 1)  # causal or anticausal?

    def transform(self, x_tr, y_tr=None):
        """ Featurize the data with the randomized coefficients

        :param x_tr: Inputdata
        :param y_tr: Targets
        :type x_tr: pandas.DataFrame
        :type y_tr: pandas.DataFrame
        :return: Featurized data
        """

        def featurize_row(row, reverse=False):
            x = scale(row['A'])[:, np.newaxis]
            y = scale(row['B'])[:, np.newaxis]
            if reverse:
                x, y = y, x
            d = np.hstack((f1(x, self.wx).mean(0), f1(y, self.wy).mean(0), f1(np.hstack((x, y)), self.wz).mean(0)))
            return d

        def featurize(data):
            ft_data = []
            ft_data_rev = []
            for idx, row in data.iterrows():
                ft_data.append(featurize_row(row))
                ft_data_rev.append(featurize_row(row, reverse=True))
            return np.vstack((np.array(ft_data), np.array(ft_data_rev)))

        x_tr = featurize(x_tr)
        print(x_tr.shape)
        x_ab, y_ab = None, None
        if y_tr:
            y_tr = y_tr['Target'].as_matrix()
            y_tr = np.hstack((y_tr, -y_tr))

            x_ab = x_tr[(y_tr == 1) | (y_tr == -1)]
            y_ab = y_tr[(y_tr == 1) | (y_tr == -1)]

        print('Featurize Finished !')

        return x_tr, y_tr, x_ab, y_ab

    def predict_proba(self, x_te):
        """ Predict causal directions of a dataset. With input data as (X,Y):
            -1 is Y->X
             1 is X->Y

        :param x_te: Inputdata
        :type x_te: pandas.DataFrame
        :return: Array containing probabilities of predictions
        :rtype: numpy.ndarray
        """
        if not self.clf0 :
            print('Model has to be trained before doing any predictions')
            raise ValueError

        x_te, _, _, _ = self.transform(x_te)
        p_te = self.clf0.predict_proba(x_te)[:, 0] * (2 * self.clf1.predict_proba(x_te)[:, 0] - 1)
        p_te = (p_te[:len(p_te)//2] - p_te[len(p_te)//2:])/2
        return p_te