"""Randomized Causation Coefficient Model.

Author : David Lopez-Paz
Ref : Lopez-Paz, David and Muandet, Krikamol and Schölkopf, Bernhard and Tolstikhin, Ilya O,
     "Towards a Learning Theory of Cause-Effect Inference", ICML 2015.
"""

from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier as CLF
from ...utils.Settings import SETTINGS
import pandas
import numpy as np
from .model import PairwiseModel

#
# def rp(k, s, d):
#     return np.hstack((np.vstack([si * np.random.randn(k, d) for si in s]),
#                       2 * np.pi * np.random.rand(k * len(s), 1))).T
#
#
# def f1(x, w):
#     if len(x.shape) == 1:
#         x = x.reshape((-1, 1))
#     return np.cos(np.dot(np.hstack((x, np.ones((x.shape[0], 1)))), w))
#
#
# class RCC(PairwiseModel):
#     """Randomized Causation Coefficient model.
#
#     Ref : Lopez-Paz, David and Muandet, Krikamol and Schölkopf, Bernhard and Tolstikhin, Ilya O,
#      "Towards a Learning Theory of Cause-Effect Inference", ICML 2015.
#     """
#
#     def __init__(self, rand_coeff=333, nb_estimators=500, nb_min_leaves=20, max_depth=1, nb_jobs=None):
#         """Initialize the model w/ its parameters.
#
#         :param rand_coeff: number of randomized coefficients
#         :param nb_estimators: number of estimators
#         :param nb_min_leaves: number of min samples leaves of the estimator
#         :param max_depth: max depth of the model
#         :param n_jobs: number of jobs to be run on parallel
#         """
#         np.random.seed(0)
#         self.K = rand_coeff
#         self.E = nb_estimators
#         self.L = nb_min_leaves
#         self.n_jobs = SETTINGS.get_default(nb_jobs=nb_jobs)
#         self.max_depth = max_depth
#
#         self.params = {'random_state': 0, 'n_estimators': self.E, 'max_features': None,
#                        'max_depth': self.max_depth, 'min_samples_leaf': self.L, 'verbose': 10, 'n_jobs': self.n_jobs}
#
#         self.wx = rp(self.K, [0.15, 1.5, 15], 1)
#         self.wy = rp(self.K, [0.15, 1.5, 15], 1)
#         self.wz = rp(self.K, [0.15, 1.5, 15], 2)
#         self.clf0 = None
#         self.clf1 = None
#
#     def fit(self, x_tr, y_tr):
#         """Fit the model on pairwise data.
#
#         :param x_tr: Input data - CEPC-format DataFrame containing pairs of variables
#         :param y_tr: Targets
#         :type x_tr: pandas.DataFrame
#         :type y_tr: pandas.DataFrame
#         """
#         x_tr, y_tr, x_ab, y_ab = self.transform(x_tr, y_tr)
#         self.fit_ftdata(x_tr, y_tr, x_ab, y_ab)
#
#     def fit_ftdata(self, x_ft, y_ft, x_ab, y_ab):
#         """Fit the model with featurized data as input.
#
#         :param x_ft: x_featurized
#         :param y_ft: y_featurized
#         :param x_ab: x_inverse_featurized
#         :param y_ab: y_inverse_featurized
#         """
#         self.clf0 = CLF(**self.params).fit(x_ft, y_ft != 0)  # causal or confounded?
#         self.clf1 = CLF(**self.params).fit(x_ab, y_ab == 1)  # causal or anticausal?
#
#     def featurize_row(self, row, reverse=False):
#         x = scale(row['A'])[:, np.newaxis]
#         y = scale(row['B'])[:, np.newaxis]
#         if reverse:
#             x, y = y, x
#         d = np.hstack((f1(x, self.wx).mean(0), f1(y, self.wy).mean(0), f1(np.hstack((x, y)), self.wz).mean(0)))
#         return d
#
#     def featurize(self, data):
#         ft_data = []
#         ft_data_rev = []
#         for idx, row in data.iterrows():
#             ft_data.append(self.featurize_row(row))
#             ft_data_rev.append(self.featurize_row(row, reverse=True))
#         return np.vstack((np.array(ft_data), np.array(ft_data_rev)))
#
#     def transform(self, x_tr, y_tr=None):
#         """Featurize the data with the randomized coefficients.
#
#         :param x_tr: Inputdata - CEPC-format DataFrame containing pairs of variables
#         :param y_tr: Targets
#         :type x_tr: pandas.DataFrame
#         :type y_tr: pandas.DataFrame or array
#         :return: Featurized data
#         """
#         x_tr = self.featurize(x_tr)
#         # print(x_tr.shape)
#         x_ab, y_ab = None, None
#         if y_tr is not None:
#             if type(y_tr) == pandas.DataFrame:
#                 y_tr = y_tr['Target'].as_matrix()
#             y_tr = np.hstack((y_tr, -y_tr))
#
#             x_ab = x_tr[(y_tr == 1) | (y_tr == -1)]
#             y_ab = y_tr[(y_tr == 1) | (y_tr == -1)]
#
#         print('Featurize Finished !')
#
#         return x_tr, y_tr, x_ab, y_ab
#
#     def predict_dataset(self, x_te):
#         """Predict causal directions of a dataset. With input data as (X,Y).
#
#             -1 is Y->X
#              1 is X->Y
#              # 0 is independent/confounding
#
#         :param x_te: Inputdata - CEPC-format DataFrame containing pairs of variables
#         :type x_te: pandas.DataFrame
#         :return: Array containing probabilities of predictions
#         :rtype: numpy.ndarray
#         """
#         if not self.clf0:
#             print('Model has to be trained before doing any predictions')
#             raise ValueError
#
#         x_te, _, _, _ = self.transform(x_te)
#         p_te = self.clf0.predict_dataset(x_te)[:, 0] * (2 * self.clf1.predict_dataset(x_te)[:, 0] - 1)
#         p_te = (p_te[:len(p_te)//2] - p_te[len(p_te)//2:])/2
#         return p_te
#
#     def predict_proba(self, a, b, **kwargs):
#         """Infer causal directions using the trained RCC model.
#
#         :param a: Variable 1
#         :param b: Variable 2
#         :return: probability (Value : 1 if a->b and -1 if b->a)
#         :rtype: float
#         """
#         if not self.clf0:
#             print('Model has to be trained before doing any predictions')
#             raise ValueError
#         a = scale(a)
#         b = scale(b)
#
#         d = np.hstack((f1(a, self.wx).mean(0), f1(b, self.wy).mean(0), f1(np.hstack((a, b)), self.wz).mean(0)))
#         p_te = self.clf0.predict_dataset(d)[:, 0] * (2 * self.clf1.predict_dataset(d)[:, 0] - 1)
#         return p_te


class RCC(PairwiseModel):
    """Randomized Causation Coefficient model.

    Ref : Lopez-Paz, David and Muandet, Krikamol and Schölkopf, Bernhard and Tolstikhin, Ilya O,
     "Towards a Learning Theory of Cause-Effect Inference", ICML 2015.
    """

    def __init__(self, rand_coeff=333, nb_estimators=500, nb_min_leaves=20, max_depth=None, s=10, nb_jobs=None, verbose=None):
        """Initialize the model w/ its parameters.

        :param rand_coeff: number of randomized coefficients
        :param nb_estimators: number of estimators
        :param nb_min_leaves: number of min samples leaves of the estimator
        :param max_depth: max depth of the model
        :param s: scaling
        :param n_jobs: number of jobs to be run on parallel
        """
        np.random.seed(0)
        self.K = rand_coeff
        self.E = nb_estimators
        self.L = nb_min_leaves
        self.n_jobs, self.verbose = SETTINGS.get_default(('nb_jobs', nb_jobs), ('verbose', verbose))
        self.max_depth = max_depth

        self.W = np.hstack((s * np.random.randn(self.K, 2),
                            2 * np.pi * np.random.rand(self.K, 1)))
        self.W2 = np.hstack((s * np.random.randn(self.K, 1),
                             2 * np.pi * np.random.rand(self.K, 1)))
        self.clf = None

    def featurize_row(self, x, y):
        x = x.ravel()
        y = y.ravel()
        b = np.ones(x.shape)
        dx = np.cos(np.dot(self.W2, np.vstack((x, b)))).mean(1)
        dy = np.cos(np.dot(self.W2, np.vstack((y, b)))).mean(1)
        if(sum(dx) > sum(dy)):
            return np.hstack((dx, dy,
                              np.cos(np.dot(self.W, np.vstack((x, y, b)))).mean(1)))
        else:
            return np.hstack((dx, dy,
                              np.cos(np.dot(self.W, np.vstack((y, x, b)))).mean(1)))

    def fit(self, x, y):
        """Train the model.

        args:
            x: pandas.Dataframe of the data
            y: targets
        """
        train = np.vstack((np.array([self.featurize_row(row.iloc[0],
                                                        row.iloc[1]) for idx, row in x.iterrows()]),
                           np.array([self.featurize_row(row.iloc[1],
                                                        row.iloc[0]) for idx, row in x.iterrows()])))
        labels = np.vstack((y, -y)).ravel()
        verbose = 1 if self.verbose else 0
        self.clf = CLF(verbose=verbose,
                       min_samples_leaf=self.L,
                       n_estimators=self.E,
                       max_depth=self.max_depth,
                       n_jobs=self.n_jobs).fit(train, labels)

    def predict_proba(self, x, y=None, **kwargs):
        if self.clf is None:
            raise ValueError("Model has to be trained before making predictions.")
        if x is pandas.Series:
            input_ = self.featurize_row(x.iloc[0], x.iloc[1]).reshape((1, -1))
        elif x is pandas.DataFrame:
            input_ = np.array([self.featurize_row(x.iloc[0], x.iloc[1]) for row in x])
        elif y is not None:
            input_ = self.featurize_row(x, y).reshape((1, -1))
        else:
            raise TypeError("DataType not understood.")
        return self.clf.predict(input_)

    def predict_dataset(self, data, **kwargs):
        return self.predict_proba(data, **kwargs)
