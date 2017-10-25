from ...utils.Settings import SETTINGS, CGNN_SETTINGS
from sklearn.preprocessing import scale
import warnings
import sys
import numpy as np
import pandas as pd
import torch as th
from torch.autograd import Variable
from joblib import Parallel, delayed
from copy import deepcopy
from .model import GraphModel
from ..pairwise.GNN import GNN
from ...utils.Loss import MMD_loss_tf, MMD_loss_th, Fourier_MMD_Loss_tf, TTestCriterion
from ...utils.Settings import SETTINGS, CGNN_SETTINGS
from ...utils.Formats import reshape_data


class CGNN_th(th.nn.Module):
    """ Generate all variables in the graph at once, torch model

    """

    def __init__(self, graph, n, **kwargs):
        """ Initialize the model, build the computation graph

        :param graph: graph to model
        :param N: Number of examples to generate
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_dim) Number of units in the hidden layer
        """
        super(CGNN_th, self).__init__()
        h_layer_dim = kwargs.get('h_layer_dim', CGNN_SETTINGS.h_layer_dim)

        self.graph = graph
        # building the computation graph
        self.graph_variables = []
        self.layers_in = []
        self.layers_out = []
        self.N = n
        self.activation = th.nn.ReLU()
        nodes = self.graph.list_nodes()
        while len(self.graph_variables) < len(nodes):
            for var in nodes:
                par = self.graph.parents(var)

                if var not in self.graph_variables and set(par).issubset(self.graph_variables):
                    # Variable can be generated
                    self.layers_in.append(
                        th.nn.Linear(len(par) + 1, h_layer_dim))
                    self.layers_out.append(th.nn.Linear(h_layer_dim, 1))
                    self.graph_variables.append(var)
                    self.add_module('linear_{}_in'.format(
                        var), th.nn.Linear(len(par) + 1, h_layer_dim))
                    self.add_module('linear_{}_out'.format(
                        var), th.nn.Linear(h_layer_dim, 1))

    def forward(self):
        """ Pass through the generative network

        :return: Generated data
        """
        generated_variables = {}
        for var in self.graph_variables:
            par = self.graph.parents(var)
            if len(par) > 0:
                inputx = th.cat([th.cat([generated_variables[parent] for parent in par], 1),
                                 Variable(th.FloatTensor(self.N, 1).normal_())], 1)
            else:
                inputx = Variable(th.FloatTensor(self.N, 1).normal_())

            generated_variables[var] = getattr(self, 'linear_{}_out'.format(var))(self.activation(getattr(
                self, 'linear_{}_in'.format(var))(inputx)))

        output = []
        for v in self.graph.list_nodes():
            output.append(generated_variables[v])

        return th.cat(output, 1)


def run_CGNN_th(df_data, graph, idx=0, run=0, verbose=True, **kwargs):
    """ Run the CGNN graph with the torch backend

    :param df_data: data DataFrame
    :param graph: graph
    :param idx: idx of the pair
    :param run: number of the run
    :param verbose: verbose
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: nb_gpu=(SETTINGS.NB_GPU) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.GPU_OFFSET) number of gpu offsets
    :param kwargs: train_epochs=(CGNN_SETTINGS.train_epochs) number of train epochs
    :param kwargs: test_epochs=(CGNN_SETTINGS.test_epochs) number of test epochs
    :param kwargs: learning_rate=(CGNN_SETTINGS.learning_rate) learning rate of the optimizer
    :return: MMD loss value of the given structure after training

    """

    gpu = kwargs.get('gpu', SETTINGS.GPU)
    nb_gpu = kwargs.get('nb_gpu', SETTINGS.NB_GPU)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.GPU_OFFSET)
    train_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.train_epochs)
    test_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.test_epochs)
    learning_rate = kwargs.get('learning_rate', CGNN_SETTINGS.learning_rate)

    list_nodes = graph.list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')
    model = CGNN_th(graph, data.shape[0], **kwargs)
    data = Variable(th.from_numpy(data))
    criterion = MMD_loss_th(data.size()[0], cuda=gpu)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    if gpu:
        data = data.cuda(gpu_offset + run % nb_gpu)
        model = model.cuda(gpu_offset + run % nb_gpu)

    # Train
    for it in range(train_epochs):
        optimizer.zero_grad()
        out = model()
        loss = criterion(data, out)
        loss.backward()
        optimizer.step()
        if verbose and it % 30 == 0:
            if gpu:
                ploss = loss.cpu.data[0]
            else:
                ploss = loss.data[0]
            print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(idx, run, it, ploss))

    # Evaluate
    mmd = 0
    for it in range(test_epochs):
        out = model()
        loss = criterion(data, out)
        if gpu:
            mmd += loss.cpu.data[0]
        else:
            mmd += loss.data[0]

    return mmd / test_epochs
