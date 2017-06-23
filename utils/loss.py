"""
Implementation of Losses
Author : Diviyan Kalainathan & Olivier Goudet
Date : 09/03/2017
"""

import tensorflow as tf
import torch as th
from torch.autograd import Variable

bandwiths_sigma = [0.01, 0.1, 1, 5, 20, 50, 100]


def MMD_loss(xy_true, xy_pred, low_memory_version=False):

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

    def forward(self, var_input, var_pred, var_true = None):

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
            lossMMD = th.cuda.FloatTensor([0])
            # lossMMD.zero_()
            lossMMD = Variable(lossMMD)
        else:
            lossMMD = Variable(th.zeros(1))
        for i in range(len(self.bandwiths)):
            kernel_val = exponent.mul(1. / self.bandwiths[i]).exp()
            lossMMD.add_((self.S.mul(kernel_val)).sum())

        return lossMMD.sqrt()


def MomentMatchingLoss_tf(xy_true, xy_pred, nb_moment = 1):
    loss = 0
    for i in range(1,nb_moment):
        mean_pred = tf.reduce_mean(xy_pred**i,0)
        mean_true = tf.reduce_mean(xy_true**i,0)
        loss += tf.sqrt(tf.reduce_sum((mean_true - mean_pred)**2))  # L2

    return loss


class MomentMatchingLoss_th(th.nn.Module):
    def __init__(self, n_moments=1):
        super(MomentMatchingLoss_th, self).__init__()
        self.moments = n_moments

    def forward(self, pred, target):
        mean_pred = th.mean(pred).expand_as(pred)
        mean_target = th.mean(target).expand_as(target)
        # std_pred = th.std(pred)
        # std_target = th.std(target)

        loss = Variable(th.FloatTensor([0]))

        for i in range(1, self.moments):
            loss_i = th.abs((th.mean(th.pow(pred-mean_pred, i))
                             - th.mean(th.pow(target-mean_target, i))))
            # loss_i = th.abs((th.mean(th.pow(pred-mean_pred, i))/th.pow(std_pred, i)
            #                  - th.mean(th.pow(target-mean_target, i))/th.pow(std_target, i)))
            loss.add_(loss_i)

        return loss

    def old_forward(self, pred, target):

        loss = Variable(th.FloatTensor([0]))
        for i in range(self.moments):
            mean_pred = th.mean(th.pow(pred, i)).expand_as(pred)
            mean_target = th.mean(th.pow(pred, i)).expand_as(target)
            loss.add_(th.sum(th.abs(mean_pred-th.abs(mean_target))))  # L1

        return loss
