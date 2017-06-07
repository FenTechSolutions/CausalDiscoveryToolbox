"""
SGNN : Shallow generative Neural Networks
Authors : Anonymous Author
Date : 10/05/2017
"""

import tensorflow as tf
import numpy as np
from ...utils.loss import MMD_loss_th as MMD
from ...utils.SETTINGS import CGNN_SETTINGS as SETTINGS
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
import torch as th
from torch.autograd import Variable
from .model import Pairwise_Model


def init(size):
    return tf.random_normal(shape=size, stddev=SETTINGS.init_weights)


class SGNN_tf(object):
    def __init__(self, N, run, pair, learning_rate=SETTINGS.learning_rate):
        """
        Build the tensorflow graph,
        The first column is set as the cause
        The second as the effect
        """
        self.run = run
        self.pair = pair
        self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])

        W_in = tf.Variable(init([2, SETTINGS.h_dim]))
        b_in = tf.Variable(init([SETTINGS.h_dim]))

        W_out = tf.Variable(init([SETTINGS.h_dim, 1]))
        b_out = tf.Variable(init([1]))

        theta_G = [W_in, b_in,
                   W_out, b_out]

        e = tf.random_normal([N, 1], mean=0, stddev=1)

        input = tf.concat([self.X, e], 1)
        hid = tf.nn.relu(tf.matmul(input, W_in) + b_in)
        out = tf.matmul(hid, W_out) + b_out

        self.G_dist_loss_xcausesy = MMD(tf.concat([self.X, self.Y], 1), tf.concat([self.X, out], 1))

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                                  .minimize(self.G_dist_loss_xcausesy, var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True):

        for it in range(SETTINGS.nb_epoch_train):
            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.X: data[:, [0]], self.Y: data[:, [1]]}
            )

            if verbose:
                if (it % 100 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.pair, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True):

        avg_score = 0

        for it in range(SETTINGS.nb_epoch_test):
            score = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={self.X: data[:, [0]], self.Y: data[:, [1]]})

            avg_score += score[0]

            if verbose:
                if (it % 100 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(self.pair, self.run, it, score[0]))

        tf.reset_default_graph()

        return avg_score / SETTINGS.nb_epoch_test


def tf_evalcausalscore_pairwise(df, idx, run):
    CGNN = SGNN_tf(df.shape[0], run, idx)
    CGNN.train(df)
    return CGNN.evaluate(df)


def tf_run_pair(m, idx, run):
    run_i = run
    if SETTINGS.GPU:
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + run_i % SETTINGS.num_gpu)):
            XY = tf_evalcausalscore_pairwise(m, idx, run)
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + run_i % SETTINGS.num_gpu)):
            YX = tf_evalcausalscore_pairwise(m[:, [1, 0]], idx, run)
            return [XY, YX]
    else:
        return [tf_evalcausalscore_pairwise(m, idx, run),
                tf_evalcausalscore_pairwise(np.fliplr(m), idx, run)]


def predict_tf(a, b):
    m = np.hstack((a, b))
    m = scale(m)
    m = m.astype('float32')

    result_pair = Parallel(n_jobs=SETTINGS.nb_jobs)(delayed(tf_run_pair)(
        m, 0, run) for run in range(SETTINGS.nb_run))

    score_AB = sum([runpair[0] for runpair in result_pair]) / SETTINGS.nb_run
    score_BA = sum([runpair[1] for runpair in result_pair]) / SETTINGS.nb_run

    return (score_BA - score_AB) / (score_BA + score_AB)


class SGNN_th(th.nn.Module):
    def __init__(self):
        """
        Build the Torch graph
        """
        super(SGNN_th, self).__init__()

        self.l1 = th.nn.Linear(2, SETTINGS.h_dim)
        self.l2 = th.nn.Linear(SETTINGS.h_dim, 1)
        self.act = th.nn.ReLU()
        # ToDo : Init parameters

    def forward(self, x):
        """
        Pass data through the net structure
        :param x: input data: shape (:,2)
        :type x: torch.Variable
        :return: output of the shallow net
        :rtype: torch.Variable

        """
        x = self.act(self.l1(x))
        return self.l2(x)


def run_SGNN_th(m, pair, run):
    """ Train and eval the SGNN on a pair

    :param m: Matrix containing cause at m[:,0],
              and effect at m[:,1]
    :type m: numpy.ndarray
    :param pair: Number of the pair
    :param run: Number of the run
    :return: Value of the evaluation after training
    :rtype: float
    """

    x = Variable(th.from_numpy(m[:, [0]]))
    y = Variable(th.from_numpy(m[:, [1]]))
    e = Variable(th.FloatTensor(m.shape[0], 1))
    SGNN = SGNN_th()

    if SETTINGS.GPU:
        x = x.cuda()
        y = y.cuda()
        e = e.cuda()
        SGNN = SGNN.cuda()

    criterion = MMD(m.shape[0], cuda=SETTINGS.GPU)

    optim = th.optim.Adam(SGNN.parameters(), lr=SETTINGS.learning_rate)
    running_loss = 0
    teloss = 0

    for i in range(SETTINGS.nb_epoch_train):
        optim.zero_grad()
        e.data.normal_()
        x_in = th.cat([x, e], 1)
        y_pred = SGNN(x_in)
        loss = criterion(x, y_pred, y)
        loss.backward()
        optim.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 300 == 299:  # print every 2000 mini-batches
            print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                  format(pair, run, i, running_loss))
            running_loss = 0.0

    # Evaluate
    for i in range(SETTINGS.nb_epoch_test):
        e.data.normal_()
        x_in = th.cat([x, e], 1)
        y_pred = SGNN(x_in)
        loss = criterion(x, y_pred, y)

        # print statistics
        running_loss += loss.data[0]
        teloss += running_loss
        if i % 300 == 299:  # print every 300 batches
            print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                  format(pair, run, i, running_loss))
            running_loss = 0.0

    return teloss / SETTINGS.nb_epoch_test


def th_run_instance(m, pair_idx, run):
    if SETTINGS.GPU:
        with th.cuda.device(SETTINGS.gpu_offset + run % SETTINGS.num_gpu):
            XY = run_SGNN_th(m, pair_idx, run)
        with th.cuda.device(SETTINGS.gpu_offset + run % SETTINGS.num_gpu):
            YX = run_SGNN_th(np.fliplr(m), pair_idx, run)

    else:
        XY = run_SGNN_th(m, pair_idx, run)
        YX = run_SGNN_th(m, pair_idx, run)

    return [XY, YX]


def predict_th(a, b):
    m = np.hstack((a, b))
    m = scale(m)
    m = m.astype('float32')
    result_pair = Parallel(n_jobs=SETTINGS.nb_jobs)(delayed(th_run_instance)(
        m, 0, run) for run in range(SETTINGS.nb_run))

    score_XY = np.mean([runpair[0] for runpair in result_pair])
    score_YX = np.mean([runpair[1] for runpair in result_pair])
    return (score_YX - score_XY) / (score_YX + score_XY)


class SGNN(Pairwise_Model):
    """
    Shallow Generative Neural networks, models the causal directions x->y and y->x with a 1-hidden layer neural network
    and a MMD loss. The causal direction is considered as the "best-fit" between the two directions
    """

    def __init__(self, backend="torch"):
        super(SGNN, self).__init__()
        if backend == "torch":
            self.backend = "torch"
        elif backend == "tensorflow":
            self.backend = "tensorflow"
        else:
            print('No backend known as {}'.format(backend))
            raise ValueError

    def predictor(self, a, b):
        if self.backend == "torch":
            return predict_th(a, b)
        elif self.backend == "tensorflow":
            return predict_tf(a, b)
        else:
            print('No backend defined !')
            raise ValueError
