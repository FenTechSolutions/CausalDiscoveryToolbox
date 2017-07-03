"""
CGNN_graph_model
Author : Olivier Goudet & Diviyan Kalainathan
Ref :
Date : 09/5/17
"""

import tensorflow as tf
import torch as th
from torch.autograd import Variable
import warnings
import os
import pandas as pd
from joblib import Parallel, delayed
import sys
import numpy as np
from copy import deepcopy
from .model import GraphModel
from ..pairwise_models.GNN import GNN
from ...utils.loss import MMD_loss_tf, MMD_loss_th
from ...utils.SETTINGS import CGNN_SETTINGS as SETTINGS


def init(size):
    """ Initialize a random tensor, normal(0,SETTINGS.init_weights).

    :param size: Size of the tensor
    :return: Tensor
    """
    return tf.random_normal(shape=size, stddev=SETTINGS.init_weights)


class CGNN_tf(object):
    def __init__(self, N, graph, run, idx, learning_rate=SETTINGS.learning_rate):
        """ Build the tensorflow graph of the CGNN structure

        :param N: Number of points
        :param graph: Graph to be run
        :param run: number of the run (only for print)
        :param idx: number of the idx (only for print)
        :param learning_rate: learning rate of the optimizer
        """

        self.run = run
        self.idx = idx
        list_nodes = graph.get_list_nodes()
        n_var = len(list_nodes)

        self.all_real_variables = tf.placeholder(tf.float32, shape=[None, n_var])

        generated_variables = {}
        theta_G = []

        while len(generated_variables) < n_var:
            # Need to generate all variables in the graph using its parents : possible because of the DAG structure
            for var in list_nodes:
                # Check if all parents are generated
                par = graph.get_parents(var)
                if (var not in generated_variables and
                        set(par).issubset(generated_variables)):
                    # Generate the variable
                    W_in = tf.Variable(init([len(par) + 1, SETTINGS.h_dim]))
                    b_in = tf.Variable(init([SETTINGS.h_dim]))
                    W_out = tf.Variable(init([SETTINGS.h_dim, 1]))
                    b_out = tf.Variable(init([1]))

                    input_v = [generated_variables[i] for i in par]
                    input_v.append(tf.random_normal([N, 1], mean=0, stddev=1))
                    input_v = tf.concat(input_v, 1)

                    out_v = tf.nn.relu(tf.matmul(input_v, W_in) + b_in)
                    out_v = tf.matmul(out_v, W_out) + b_out

                    generated_variables[var] = out_v
                    theta_G.extend([W_in, b_in, W_out, b_out])

        listvariablegraph = []
        for var in list_nodes:
            listvariablegraph.append(generated_variables[var])

        all_generated_variables = tf.concat(listvariablegraph, 1)

        self.G_dist_loss_xcausesy = MMD_loss_tf(self.all_real_variables, all_generated_variables)

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True):
        """ Train the initialized model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :return: None
        """
        for it in range(SETTINGS.nb_epoch_train):

            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_real_variables: data}
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.idx, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :return: mean MMD loss value of the CGNN structure on the data
        """

        sumMMD_tr = 0

        for it in range(SETTINGS.nb_epoch_test):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={
                self.all_real_variables: data})

            sumMMD_tr += MMD_tr[0]

            if verbose and it % 100 == 0:
                print('Pair:{}, Run:{}, Iter:{}, score:{}'
                          .format(self.idx, self.run, it, MMD_tr[0]))

        tf.reset_default_graph()

        return sumMMD_tr / SETTINGS.nb_epoch_test


def run_CGNN_tf(df_data, graph, idx=0, run=0):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param df_data: data corresponding to the graph
    :param graph: Graph to be run
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :return: MMD loss value of the given structure after training
    """

    list_nodes = graph.get_list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')

    if SETTINGS.GPU:
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + run % SETTINGS.num_gpu)):
            model = CGNN_tf(df_data.shape[0], graph, run, idx)
            model.train(data)
            return model.evaluate(data)
    else:
        model = CGNN(df_data.shape[0], graph, run, idx)
        model.train(data)
        return model.evaluate(data)


class CGNN_th(th.nn.Module):
    """ Generate all variables in the graph at once, torch model

    """
    def __init__(self, graph, N):
        """ Initialize the model, build the computation graph

        :param graph: graph to model
        :param N: Number of examples to generate
        """
        super(CGNN_th, self).__init__()
        self.graph = graph
        # building the computation graph
        self.graph_variables = []
        self.params_in = []
        self.params_out = []
        self.N = N
        self.activation = th.nn.ReLU()
        nodes = self.graph.get_list_nodes()
        while self.graph_variables < len(nodes):
            for var in nodes:
                par = self.graph.get_parents(var)
                if (var not in self.graph_variables and
                    set(par).issubset(self.graph_variables)):
                    # Variable can be generated
                    self.params_in.append(th.nn.Linear(len(par) + 1, SETTINGS.h_dim))
                    self.params_out.append(th.nn.Linear(SETTINGS.h_dim, 1))
                    self.graph_variables.append(par)

    def forward(self):
        """ Pass through the generative network

        :return: Generated data
        """
        generated_variables = {}
        for var, layer_in, layer_out in zip(self.graph_variables, self.params_in, self.params_out):
            par = self.graph.get_parents(var)
            if len(par) > 0:
                inputx = th.cat([th.cat([generated_variables[parent] for parent in par], 1),
                                 th.FloatTensor(self.N, 1).normal_()], 1)
            else:
                inputx = th.FloatTensor(self.N, 1).normal_()

            generated_variables[var] = layer_out(self.activation(layer_in(inputx)))

        output = []
        for v in self.graph.get_list_nodes():
            output.append(generated_variables[v])

        return th.cat(output, 1)


def run_CGNN_th(df_data, graph, idx=0, run=0, verbose=True):
    """

    :param df_data:
    :param graph:
    :param idx:
    :param run:
    :param verbose:
    :return:
    """

    list_nodes = graph.get_list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')
    model = CGNN_th(graph, data.shape[0])
    data = th.from_numpy(data)
    criterion = MMD_loss_th(data.shape[0], cuda=SETTINGS.GPU)
    optimizer = th.optim.Adam(model.params_in + model.params_out)

    if SETTINGS.GPU:
        data = data.cuda(SETTINGS.gpu_offset + run % SETTINGS.num_gpu)
        model = model.cuda(SETTINGS.gpu_offset + run % SETTINGS.num_gpu)

    # Train
    for it in range(SETTINGS.nb_epoch_train):
        out = model()
        loss = criterion(data, out)
        loss.backward()
        optimizer.step()
        if verbose and it%50 == 0:
            if SETTINGS.GPU:
                ploss=loss.cpu.data[0]
            else:
                ploss=loss.data[0]
            print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(idx, run, it, ploss))

    #Evaluate
    mmd = 0
    for it in range(SETTINGS.nb_epoch_test):
        out = model()
        loss = criterion(data, out)
        if SETTINGS.GPU:
            mmd += loss.cpu.data[0]
        else:
            mmd += loss.data[0]

    return mmd/SETTINGS.nb_epoch_test


def unpack_results(result_pairs):
    """  Process the results given by the multiprocessing loop

    :param result_pairs: List of MMD results
    :return:
    """
    run_no = 0

    sum_score_graph = 0
    for result_pair in result_pairs:

        run_no += 1
        denom = SETTINGS.nb_run

        score_graph = result_pair
        if np.isfinite(score_graph):
            sum_score_graph += score_graph
        else:
            denom = -1
            warnings.warn('NaN value', RuntimeWarning)

    return sum_score_graph / denom


def hill_climbing(graph, data, run_cgnn_function):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph:
    :param data:
    :param run_cgnn_function:
    :return:
    """
    loop = 0
    tested_configurations = [graph.get_dict_nw()]
    improvement = True
    result = []
    result_pairs = Parallel(n_jobs=SETTINGS.nb_jobs)(delayed(run_cgnn_function)(
        data, graph, 0, run) for run in range(SETTINGS.nb_run))

    score_network = unpack_results(result_pairs)
    globalscore = score_network

    print("Graph score : " + str(globalscore))

    while improvement:
        loop += 1
        improvement = False
        list_edges = graph.get_list_edges()
        for idx_pair in range(len(list_edges)):
            edge = list_edges[idx_pair]
            test_graph = deepcopy(graph)
            test_graph.reverse_edge(edge[0], edge[1])

            if (test_graph.is_cyclic()
                or test_graph.get_dict_nw() in tested_configurations):
                print('No Evaluation for {}'.format([edge]))
            else:
                print('Edge {} in evaluation :'.format(edge))
                tested_configurations.append(test_graph.get_dict_nw())
                result_pairs = Parallel(n_jobs=SETTINGS.nb_jobs)(delayed(run_cgnn_function)(
                    data, test_graph, idx_pair, run) for run in range(SETTINGS.nb_run))

                score_network = unpack_results(result_pairs)
                score_network = score_network

                print("Current score : " + str(score_network))
                print("Best score : " + str(globalscore))

                if score_network < globalscore:
                    graph.reverse_edge(edge[0], edge[1])
                    improvement = True
                    print('Edge {} got reversed !'.format(edge))
                    globalscore = score_network

    return graph


class CGNN(GraphModel):
    """
    CGNN Model ; Using generative models, generate the whole causal graph and improve causal
    direction predictions in the graph.
    """

    def __init__(self, backend='PyTorch'):
        """ Initialize the CGNN Model.

        :param backend: Choose the backend to use, either 'PyTorch' or 'TensorFlow'
        """
        super(CGNN, self).__init__()
        self.backend = backend

        if self.backend == 'TensorFlow':
            self.infer_graph = run_CGNN_tf
        elif self.backend == 'PyTorch':
            self.infer_graph = run_CGNN_th
        else:
            print('No backend known as {}'.format(self.backend))
            raise ValueError

    def create_graph_from_data(self, data):
        print("The CGNN model is not able (yet?) to model the graph directly from raw data")
        raise ValueError

    def orient_directed_graph(self, data, dag, alg='HC', log=False):
        """

        :param data:
        :param dag:
        :param alg:
        :param log: Save logs of the execution
        :return:
        """
        alg_dico = {'HC': hill_climbing}
        return alg_dico[alg](dag, data, self.infer_graph)

    def orient_undirected_graph(self, data, umg):
        """

        :param data:
        :param umg:
        :return:
        """

        warnings.warn("The pairwise GNN model is computed once on the UMG to initialize the model and start with a DAG")
        gnn = GNN(backend=self.backend)
        dag = gnn.orient_graph(data, umg)  # Pairwise method
        return self.orient_directed_graph(data, dag)
