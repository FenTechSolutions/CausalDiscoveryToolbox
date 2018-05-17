"""Implementation of Losses.

Author : Diviyan Kalainathan & Olivier Goudet
Date : 09/03/2017
"""


from .Settings import SETTINGS
import numpy as np
from scipy.stats import ttest_ind
import torch as th
from torch.autograd import Variable

bandwiths_gamma = [0.005, 0.05, 0.25, 0.5, 1, 5, 50]


class TTestCriterion(object):
    def __init__(self, max_iter, runs_per_iter, threshold=0.01):
        super(TTestCriterion, self).__init__()
        self.threshold = threshold
        self.max_iter = max_iter
        self.runs_per_iter = runs_per_iter
        self.iter = 0
        self.p_value = np.nan

    def loop(self, xy, yx):
        if len(xy) > 0:
            self.iter += self.runs_per_iter
        if self.iter < 2:
            return True
        t_test, self.p_value = ttest_ind(xy, yx, equal_var=False)
        if self.p_value > self.threshold and self.iter < self.max_iter:
            return True
        else:
            return False


class MMDloss(th.nn.Module):
    """Maximum Mean Discrepancy Metric to compare empirical distributions.

    Ref: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
    """

    def __init__(self, input_size, bandwiths=None, device=None):
        """Init the model."""
        super(MMDloss, self).__init__()
        device = SETTINGS.get_default(device=device)
        if bandwiths is None:
            self.bandwiths = [0.01, 0.1, 1, 10, 100]
        else:
            self.bandwiths = bandwidths
        s = th.cat([th.ones([input_size, 1]) / input_size,
                    th.ones([input_size, 1]) / -input_size], 0)

        self.S = s.mm(s.t()).to(device)

    def forward(self, x, y):
        """Compute the MMD statistic between x and y."""
        X = th.cat([x, y], 0)
        # dot product between all combinations of rows in 'X'
        XX = X @ X.t()
        # dot product of rows with themselves
        # Old code : X2 = (X * X).sum(dim=1)
        X2 = XX.diag().unsqueeze(0)
        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        exponent = -2*XX + X2.expand_as(XX) + X2.t().expand_as(XX)

        lossMMD = th.sum(self.S * sum([(exponent * -bandwith).exp() for bandwith in self.bandwiths]))
        return lossMMD.sqrt()


class MomentMatchingLoss_th(th.nn.Module):
    """k-moments loss, k being a parameter.

    These moments are raw moments and not normalized.
    """

    def __init__(self, n_moments=1):
        """Initialize the loss model.

        :param n_moments: number of moments
        """
        super(MomentMatchingLoss_th, self).__init__()
        self.moments = n_moments

    def forward(self, pred, target):
        """Compute the loss model.

        :param pred: predicted Variable
        :param target: Target Variable
        :return: Loss
        """
        loss = Variable(th.FloatTensor([0]))
        for i in range(1, self.moments):
            mk_pred = th.mean(th.pow(pred, i), 0)
            mk_tar = th.mean(th.pow(target, i), 0)

            loss.add_(th.mean((mk_pred - mk_tar) ** 2))  # L2

        return loss
