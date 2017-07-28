"""
Implementation of Losses
Author : Diviyan Kalainathan & Olivier Goudet
Date : 09/03/2017
"""

import tensorflow as tf
import torch as th
from torch.autograd import Variable

bandwiths_sigma = [0.01, 0.1, 1, 5, 20, 50, 100]


def MMD_loss_tf(xy_true, xy_pred, low_memory_version=False):

    N, _ = xy_pred.get_shape().as_list()

    if (low_memory_version == 1):

        loss = 0.0

        XX = tf.matmul(xy_pred, tf.transpose(xy_pred))
        X2 = tf.reduce_sum(xy_pred * xy_pred, 1, keep_dims=True)
        exponentXX = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)
        sXX = tf.constant(1.0 / N ** 2, shape=[N, 1])

        for i in range(len(bandwiths_sigma)):
            kernel_val = tf.exp(1.0 / bandwiths_sigma[i] * exponentXX)
            loss += tf.reduce_sum(sXX * kernel_val)

        YY = tf.matmul(xy_true, tf.transpose(xy_true))
        Y2 = tf.reduce_sum(xy_true * xy_true, 1, keep_dims=True)
        exponentYY = YY - 0.5 * Y2 - 0.5 * tf.transpose(Y2)
        sYY = tf.constant(1.0 / N ** 2, shape=[N, 1])

        for i in range(len(bandwiths_sigma)):
            kernel_val = tf.exp(1.0 / bandwiths_sigma[i] * exponentYY)
            loss += tf.reduce_sum(sYY * kernel_val)

        XY = tf.matmul(xy_pred, tf.transpose(xy_true))
        exponentXY = XY - 0.5 * X2 - 0.5 * tf.transpose(Y2)
        sXY = -tf.constant(2.0 / N ** 2, shape=[N, 1])

        for i in range(len(bandwiths_sigma)):
            kernel_val = tf.exp(1.0 / bandwiths_sigma[i] * exponentXY)
            loss += tf.reduce_sum(sXY * kernel_val)

    else:

        X = tf.concat([xy_pred, xy_true], 0)
        XX = tf.matmul(X, tf.transpose(X))
        X2 = tf.reduce_sum(X * X, 1, keep_dims=True)
        exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)

        s1 = tf.constant(1.0 / N, shape=[N, 1])
        s2 = -tf.constant(1.0 / N, shape=[N, 1])
        s = tf.concat([s1, s2], 0)
        S = tf.matmul(s, tf.transpose(s))

        loss = 0

        for i in range(len(bandwiths_sigma)):
            kernel_val = tf.exp(1.0 / bandwiths_sigma[i] * exponent)
            loss += tf.reduce_sum(S * kernel_val)

    return tf.sqrt(loss)


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


def MMD_Fourrier_loss_tf(xy_true, xy_pred):

    N, nDim = xy_pred.get_shape().as_list()

    nBasis = 1000

    k0 = 1

    W = tf.random_normal([nBasis, nDim], mean=0, stddev=1)

    tx = tf.matmul(W, tf.transpose(xy_true))
    ty = tf.matmul(W, tf.transpose(xy_pred))

    loss = 0

    for kk in range(len(bandwiths_sigma)):

        sgm = bandwiths_sigma[kk];
        t1 = tx / sgm;
        t2 = tf.concat([tf.cos(t1), tf.sin(t1)],0)

        phiPos = tf.reduce_sum(t2, axis=1)

        t1 = ty / sgm;
        t2 = tf.concat([tf.cos(t1), tf.sin(t1)], 0)
        phiNeg = tf.reduce_sum(t2, axis=1)

        phiPos = phiPos / N / np.sqrt(nBasis)
        phiNeg = phiNeg / N / np.sqrt(nBasis)

        # t = tf.reduce_sum((phiPos - phiNeg) ** 2) + tf.reduce_sum(phiPos ** 2) / (N - 1) + tf.reduce_sum(phiNeg ** 2) / (
        # N - 1) - k0 * (N + N - 2) / (N - 1) / (N - 1)

        t = tf.reduce_sum((phiPos - phiNeg) ** 2) + tf.reduce_sum(phiPos ** 2) / (N - 1) + tf.reduce_sum(phiNeg ** 2) / (
        N - 1)

        # loss += tf.sqrt((tf.maximum(t, 0)))
        loss += tf.sqrt(t)

    return loss


def rp(k,s,d):

  return tf.transpose(tf.concat([tf.concat([si*tf.random_normal([k,d], mean=0, stddev=1) for si in s], axis = 0), 2*np.pi*tf.random_normal([k*len(s),1], mean=0, stddev=1)], axis = 1))

def f1(x,wz,N):

  ones = tf.ones((N, 1))
  x_ones = tf.concat([x, ones], axis = 1)
  mult = tf.matmul(x_ones,wz)

  return tf.cos(mult)

def David_MMD_Loss_tf(xy_true, xy_pred):

  N, nDim = xy_pred.get_shape().as_list()

  Ka = 100

  wz = rp(Ka, bandwiths_sigma, nDim)

  e1 = tf.reduce_mean(f1(xy_true,wz,N), axis =0)
  e2 = tf.reduce_mean(f1(xy_pred,wz,N), axis =0)

  return tf.reduce_mean((e1 - e2)**2)


def MMD_Linear_loss_tf(x, y):

    N_, nDim = y.get_shape().as_list()

    N = int(N_/2)

    idx_pair = [2*i  for i in range(N)]
    idx_impair = [2*i +1 for i in range(N)]

    xx = tf.gather(x,idx_pair) - tf.gather(x,idx_impair)
    yy = tf.gather(y,idx_pair) - tf.gather(y,idx_impair)
    xy = tf.gather(x, idx_pair) - tf.gather(y, idx_impair)
    yx = tf.gather(x, idx_impair) - tf.gather(y, idx_pair)

    loss = 0

    for nSg in bandwiths_sigma:

        exp_xx = tf.exp(-tf.reduce_sum(xx**2, axis=1)*nSg)
        exp_yy = tf.exp(-tf.reduce_sum(yy**2, axis=1)*nSg)
        exp_xy = tf.exp(-tf.reduce_sum(xy**2, axis=1)*nSg)
        exp_yx = tf.exp(-tf.reduce_sum(yx**2, axis=1)*nSg)

        d = tf.reduce_sum(exp_xx + exp_yy - exp_xy - exp_yx, axis=0)
        loss += tf.sqrt(tf.maximum(d,0))

    return loss


def MomentMatchingLoss_tf(xy_true, xy_pred, nb_moment = 1):
    """ k-moments loss, k being a parameter. These moments are raw moments and not normalized

    """

    loss = 0
    for i in range(1, nb_moment):
        mean_pred = tf.reduce_mean(xy_pred**i, 0)
        mean_true = tf.reduce_mean(xy_true**i, 0)
        loss += tf.sqrt(tf.reduce_sum((mean_true - mean_pred)**2))  # L2

    return loss


class MomentMatchingLoss_th(th.nn.Module):
    """ k-moments loss, k being a parameter. These moments are raw moments and not normalized

    """
    def __init__(self, n_moments=1):
        """ Initialize the loss model

        :param n_moments: number of moments
        """
        super(MomentMatchingLoss_th, self).__init__()
        self.moments = n_moments

    def forward(self, pred, target):
        """ Compute the loss model

        :param pred: predicted Variable
        :param target: Target Variable
        :return: Loss
        """

        loss = Variable(th.FloatTensor([0]))
        for i in range(1, self.moments):
            mk_pred = th.mean(th.pow(pred, i), 0)
            mk_tar = th.mean(th.pow(target, i), 0)

            loss.add_(th.mean((mk_pred - mk_tar)**2))  # L2

        return loss
