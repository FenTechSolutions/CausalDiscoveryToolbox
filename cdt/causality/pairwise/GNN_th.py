import numpy as np
from ...utils.Loss import MMD_loss_th as MMD_th
from ...utils.Loss import TTestCriterion
from ...utils.Graph import DirectedGraph
from ...utils.Settings import SETTINGS, CGNN_SETTINGS
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
import torch as th
from torch.autograd import Variable
from .model import PairwiseModel
from pandas import DataFrame
from ...utils.Formats import reshape_data


class GNN_th(th.nn.Module):
    def __init__(self, **kwargs):
        """
        Build the Torch graph
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim) Number of units in the hidden layer
        """
        super(GNN_th, self).__init__()
        h_layer_dim = kwargs.get('h_layer_dim', CGNN_SETTINGS.h_layer_dim)
        self.s1 = th.nn.Linear(1, h_layer_dim)
        self.s2 = th.nn.Linear(h_layer_dim, 1)

        self.l1 = th.nn.Linear(2, h_layer_dim)
        self.l2 = th.nn.Linear(h_layer_dim, 1)
        self.act = th.nn.ReLU()
        # ToDo : Init parameters

    def forward(self, x1, x2):
        """
        Pass data through the net structure
        :param x: input data: shape (:,2)
        :type x: torch.Variable
        :return: output of the shallow net
        :rtype: torch.Variable

        """
        x = self.s2(self.act(self.s1(x1)))
        y = self.act(self.l1(th.cat([x, x2], 1)))
        return x, self.l2(y)


def run_GNN_th(m, pair, run, **kwargs):
    """ Train and eval the GNN on a pair

    :param m: Matrix containing cause at m[:,0],
              and effect at m[:,1]
    :type m: numpy.ndarray
    :param pair: Number of the pair
    :param run: Number of the run
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: train_epochs=(CGNN_SETTINGS.nb_epoch_train) number of train epochs
    :param kwargs: test_epochs=(CGNN_SETTINGS.nb_epoch_test) number of test epochs
    :param kwargs: learning_rate=(CGNN_SETTINGS.learning_rate) learning rate of the optimizer
    :return: Value of the evaluation after training
    :rtype: float
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    train_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.train_epochs)
    test_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.test_epochs)
    learning_rate = kwargs.get('learning_rate', CGNN_SETTINGS.learning_rate)

    target = Variable(th.from_numpy(m))
    # x = Variable(th.from_numpy(m[:, [0]]))
    # y = Variable(th.from_numpy(m[:, [1]]))
    e = Variable(th.FloatTensor(m.shape[0], 1))
    es = Variable(th.FloatTensor(m.shape[0], 1))
    GNN = GNN_th(**kwargs)

    if gpu:
        target = target.cuda()
        e = e.cuda()
        es = es.cuda()
        GNN = GNN.cuda()

    criterion = MMD_th(m.shape[0], cuda=gpu)

    optim = th.optim.Adam(GNN.parameters(), lr=learning_rate)
    running_loss = 0
    teloss = 0

    for i in range(train_epochs):
        optim.zero_grad()
        e.data.normal_()
        es.data.normal_()
        pred = GNN(es, e)

        loss = criterion(target, th.cat(pred, 1))
        loss.backward()
        optim.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 300 == 299:
            print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                  format(pair, run, i, running_loss))
            running_loss = 0.0

    # Evaluate
    for i in range(test_epochs):
        e.data.normal_()
        es.data.normal_()
        pred = GNN(es, e)
        loss = criterion(target, th.cat(pred, 1))

        # print statistics
        running_loss += loss.data[0]
        teloss += running_loss
        if i % 300 == 299:  # print every 300 batches
            print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                  format(pair, run, i, running_loss))
            running_loss = 0.0

    return teloss / test_epochs


def th_run_instance(m, pair_idx=0, run=0, **kwargs):
    """

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param pair_idx: print purposes
    :param run: numner of the run (for GPU dispatch)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: nb_gpu=(SETTINGS.NB_GPU) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.GPU_OFFSET) number of gpu offsets
    :return:
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    nb_gpu = kwargs.get('nb_gpu', SETTINGS.NB_GPU)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.GPU_OFFSET)

    if gpu:
        with th.cuda.device(gpu_offset + run % nb_gpu):
            XY = run_GNN_th(m, pair_idx, run, **kwargs)
        with th.cuda.device(gpu_offset + run % nb_gpu):
            # fliplr is unsupported in Torch
            YX = run_GNN_th(m[:, [1, 0]], pair_idx, run, **kwargs)

    else:
        XY = run_GNN_th(m, pair_idx, run, **kwargs)
        YX = run_GNN_th(m, pair_idx, run, **kwargs)

    return [XY, YX]
