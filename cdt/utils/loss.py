"""Pytorch implementation of Losses and tools."""
from .Settings import SETTINGS
import numpy as np
from scipy.stats import ttest_ind
import torch as th


class TTestCriterion(object):
    """ A loop criterion based on t-test to check significance of results.

    Args:
        max_iter (int): Maximum number of iterations authorized
        runs_per_iter (int): Number of runs performed per iteration
        threshold (float): p-value threshold, under which the loop is stopped.
    """
    def __init__(self, max_iter, runs_per_iter, threshold=0.01):
        super(TTestCriterion, self).__init__()
        self.threshold = threshold
        self.max_iter = max_iter
        self.runs_per_iter = runs_per_iter
        self.iter = 0
        self.p_value = np.nan

    def loop(self, xy, yx):
        """ Tests the loop condition based on the new results and the
        parameters.

        Args:
            xy (list): list containing all the results for one set of samples
            yx (list): list containing all the results for the other set.

        Returns:
            bool: True if the loop has to continue, False otherwise.
        """
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
    """"**[torch.nn.Module]** Maximum Mean Discrepancy Metric to compare
    empirical distributions.

    The MMD score is defined by:

    .. math::
        \\widehat{MMD_k}(\\mathcal{D}, \\widehat{\\mathcal{D}}) = 
        \\frac{1}{n^2} \\sum_{i, j = 1}^{n} k(x_i, x_j) + \\frac{1}{n^2}
        \\sum_{i, j = 1}^{n} k(\\hat{x}_i, \\hat{x}_j) - \\frac{2}{n^2} 
        \\sum_{i,j = 1}^n k(x_i, \\hat{x}_j)

    where :math:`\\mathcal{D} \\text{ and } \\widehat{\\mathcal{D}}` represent 
    respectively the observed and empirical distributions, :math:`k` represents
    the RBF kernel and :math:`n` the batch size.

    Args:
        input_size (int): Fixed batch size.
        bandwiths (list): List of bandwiths to take account of. Defaults at
            [0.01, 0.1, 1, 10, 100]
        device (str): PyTorch device on which the computation will be made.
            Defaults at ``cdt.SETTINGS.default_device``.

    Inputs: empirical, observed
        Forward pass: Takes both the true samples and the generated sample in any order 
        and returns the MMD score between the two empirical distributions.

        + **empirical** distribution of shape `(batch_size, features)`: torch.Tensor
          containing the empirical distribution
        + **observed** distribution of shape `(batch_size, features)`: torch.Tensor
          containing the observed distribution.

    Outputs: score
        + **score** of shape `(1)`: Torch.Tensor containing the loss value.

    .. note::
        Ref: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, 
        B., & Smola, A. (2012). A kernel two-sample test.
        Journal of Machine Learning Research, 13(Mar), 723-773.
    """

    def __init__(self, input_size, bandwidths=None, device=None):
        """Init the model."""
        super(MMDloss, self).__init__()
        device = SETTINGS.get_default(device=device)
        if bandwidths is None:
            self.bandwidths = [0.01, 0.1, 1, 10, 100]
        else:
            self.bandwidths = bandwidths
        s = th.cat([th.ones([input_size, 1]) / input_size,
                    th.ones([input_size, 1]) / -input_size], 0)

        self.S = (s @ s.t()).to(device)

    def forward(self, x, y):
        X = th.cat([x, y], 0)
        # dot product between all combinations of rows in 'X'
        XX = X @ X.t()
        # dot product of rows with themselves
        # Old code : X2 = (X * X).sum(dim=1)
        # X2 = XX.diag().unsqueeze(0)
        X2 = (X * X).sum(dim=1).unsqueeze(0)
        # print(X2.shape)
        # raise ValueError
        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        exponent = -2*XX + X2.expand_as(XX) + X2.t().expand_as(XX)

        lossMMD = th.sum(sum([self.S *(exponent * -bandwidth).exp() 
                              for bandwidth in self.bandwidths]))
        return lossMMD


class MomentMatchingLoss(th.nn.Module):
    """**[torch.nn.Module]** L2 Loss between k-moments between two
    distributions, k being a parameter.

    These moments are raw moments and not normalized.
    The loss is an L2 loss between the moments:

    .. math::
        MML(X, Y) = \\sum_{m=1}^{m^*} \\left( \\frac{1}{n_x} \\sum_{i=1}^{n_x} {x_i}^m 
        - \\frac{1}{n_y} \\sum_{j=1}^{n_y} {y_j}^m \\right)^2

    where :math:`m^*` represent the number of moments to compute.

    Args:
        n_moments (int): Number of moments to compute.

    Input: (X, Y)
        + **X** represents the first empirical distribution in a torch.Tensor of
          shape `(?, features)`
        + **Y** represents the second empirical distribution in a torch.Tensor of
          shape `(?, features)`

    Output: mml
        + **mml** is the output of the forward pass and is differenciable. 
          torch.Tensor of shape `(1)`
    """

    def __init__(self, n_moments=1):
        """Initialize the loss model.

        :param n_moments: number of moments
        """
        super(MomentMatchingLoss, self).__init__()
        self.moments = n_moments

    def forward(self, pred, target):
        """Compute the loss model.

        :param pred: predicted Variable
        :param target: Target Variable
        :return: Loss
        """
        loss = th.FloatTensor([0])
        for i in range(1, self.moments):
            mk_pred = th.mean(th.pow(pred, i), 0)
            mk_tar = th.mean(th.pow(target, i), 0)

            loss.add_(th.mean((mk_pred - mk_tar) ** 2))  # L2

        return loss
