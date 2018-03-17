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


class MMD_loss_th(th.nn.Module):
    def __init__(self, input_size, cuda=False):
        super(MMD_loss_th, self).__init__()
        self.bandwiths = [0.01, 0.1, 1, 5, 20, 50, 100]
        self.cuda = cuda
        if self.cuda:
            s1 = th.cuda.FloatTensor(input_size, 1).fill_(1)
            s2 = s1.clone()
            s = th.cat([s1.div(input_size),
                        s2.div(-input_size)], 0)

        else:
            s = th.cat([(th.ones([input_size, 1])).div(input_size),
                        (th.ones([input_size, 1])).div(-input_size)], 0)

        self.S = s.mm(s.t())
        self.S = Variable(self.S)

    def forward(self, var_input, var_pred, var_true=None):

        # MMD Loss
        if var_true is None:
            X = th.cat([var_input, var_pred], 0)
        else:
            X = th.cat([th.cat([var_input, var_pred], 1),
                        th.cat([var_input, var_true], 1)], 0)
        # dot product between all combinations of rows in 'X'
        XX = X.mm(X.t())

        # dot product of rows with themselves
        X2 = (X.mul(X)).sum(dim=1)

        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
        exponent = XX.sub((X2.mul(0.5)).expand_as(XX)) - \
            (((X2.t()).mul(0.5)).expand_as(XX))

        if self.cuda:
            lossMMD = Variable(th.cuda.FloatTensor([0]))
        else:
            lossMMD = Variable(th.zeros(1))
        for i in range(len(self.bandwiths)):
            kernel_val = exponent.mul(1. / self.bandwiths[i]).exp()
            lossMMD.add_((self.S.mul(kernel_val)).sum())

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
