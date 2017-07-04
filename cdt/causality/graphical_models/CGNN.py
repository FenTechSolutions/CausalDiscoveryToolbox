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
from joblib import Parallel, delayed
import sys
import numpy as np
from copy import deepcopy
from .model import GraphModel
from ..pairwise_models.GNN import GNN
from ...utils.Loss import MMD_loss_tf, MMD_loss_th
from ...utils.Settings import Settings as SETTINGS


def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class CGNN_tf(object):
    def __init__(self, N, graph, run, idx, **kwargs):
        """ Build the tensorflow graph of the CGNN structure

        :param N: Number of points
        :param graph: Graph to be run
        :param run: number of the run (only for print)
        :param idx: number of the idx (only for print)
        :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
        :param kwargs: h_layer_dim=(SETTINGS.h_dim) Number of units in the hidden layer
        """
        learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)
        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_dim)
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
                    W_in = tf.Variable(init([len(par) + 1, h_layer_dim], **kwargs))
                    b_in = tf.Variable(init([h_layer_dim], **kwargs))
                    W_out = tf.Variable(init([h_layer_dim, 1], **kwargs))
                    b_out = tf.Variable(init([1], **kwargs))

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

    def train(self, data, verbose=True, **kwargs):
        """ Train the initialized model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: train_epochs=(SETTINGS.nb_epoch_train) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', SETTINGS.nb_epoch_train)
        for it in range(train_epochs):

            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_real_variables: data}
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.idx, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True, **kwargs):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: test_epochs=(SETTINGS.nb_epoch_test) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', SETTINGS.nb_epoch_test)
        sumMMD_tr = 0

        for it in range(test_epochs):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={
                self.all_real_variables: data})

            sumMMD_tr += MMD_tr[0]

            if verbose and it % 100 == 0:
                print('Pair:{}, Run:{}, Iter:{}, score:{}'
                          .format(self.idx, self.run, it, MMD_tr[0]))

        tf.reset_default_graph()

        return sumMMD_tr / test_epochs


def run_CGNN_tf(df_data, graph, idx=0, run=0, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param df_data: data corresponding to the graph
    :param graph: Graph to be run
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: num_gpu=(SETTINGS.num_gpu) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.gpu_offset) number of gpu offsets
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    num_gpu = kwargs.get('num_gpu', SETTINGS.num_gpu)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.gpu_offset)

    list_nodes = graph.get_list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')

    if gpu:
        with tf.device('/gpu:' + str(gpu_offset + run % num_gpu)):
            model = CGNN_tf(df_data.shape[0], graph, run, idx, **kwargs)
            model.train(data, **kwargs)
            return model.evaluate(data, **kwargs)
    else:
        model = CGNN_tf(df_data.shape[0], graph, run, idx, **kwargs)
        model.train(data, **kwargs)
        return model.evaluate(data, **kwargs)


class CGNN_th(th.nn.Module):
    """ Generate all variables in the graph at once, torch model

    """
    def __init__(self, graph, n, **kwargs):
        """ Initialize the model, build the computation graph

        :param graph: graph to model
        :param N: Number of examples to generate
        :param kwargs: h_layer_dim=(SETTINGS.h_dim) Number of units in the hidden layer
        """
        super(CGNN_th, self).__init__()
        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_dim)

        self.graph = graph
        # building the computation graph
        self.graph_variables = []
        self.layers_in = []
        self.layers_out = []
        self.N = n
        self.activation = th.nn.ReLU()
        nodes = self.graph.get_list_nodes()
        while len(self.graph_variables) < len(nodes):
            for var in nodes:
                par = self.graph.get_parents(var)

                if var not in self.graph_variables and set(par).issubset(self.graph_variables):
                    # Variable can be generated
                    self.layers_in.append(th.nn.Linear(len(par) + 1, h_layer_dim))
                    self.layers_out.append(th.nn.Linear(h_layer_dim, 1))
                    self.graph_variables.append(var)
                    self.add_module('linear_{}_in'.format(var), th.nn.Linear(len(par) + 1, h_layer_dim))
                    self.add_module('linear_{}_out'.format(var), th.nn.Linear(h_layer_dim, 1))


    def forward(self):
        """ Pass through the generative network

        :return: Generated data
        """
        generated_variables = {}
        for var in self.graph_variables:
            par = self.graph.get_parents(var)
            if len(par) > 0:
                inputx = th.cat([th.cat([generated_variables[parent] for parent in par], 1),
                                 Variable(th.FloatTensor(self.N, 1).normal_())], 1)
            else:
                inputx = Variable(th.FloatTensor(self.N, 1).normal_())

            generated_variables[var] = getattr(self, 'linear_{}_out'.format(var))(self.activation(getattr(
                self, 'linear_{}_in'.format(var))(inputx)))

        output = []
        for v in self.graph.get_list_nodes():
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
    :param kwargs: num_gpu=(SETTINGS.num_gpu) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.gpu_offset) number of gpu offsets
    :param kwargs: train_epochs=(SETTINGS.nb_epoch_train) number of train epochs
    :param kwargs: test_epochs=(SETTINGS.nb_epoch_test) number of test epochs
    :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
    :return: MMD loss value of the given structure after training

    """

    gpu = kwargs.get('gpu', SETTINGS.GPU)
    num_gpu = kwargs.get('num_gpu', SETTINGS.num_gpu)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.gpu_offset)
    train_epochs = kwargs.get('test_epochs', SETTINGS.nb_epoch_train)
    test_epochs = kwargs.get('test_epochs', SETTINGS.nb_epoch_test)
    learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)

    list_nodes = graph.get_list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')
    model = CGNN_th(graph, data.shape[0], **kwargs)
    data = Variable(th.from_numpy(data))
    criterion = MMD_loss_th(data.size()[0], cuda=gpu)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    if gpu:
        data = data.cuda(gpu_offset + run % num_gpu)
        model = model.cuda(gpu_offset + run % num_gpu)

    # Train
    for it in range(train_epochs):
        optimizer.zero_grad()
        out = model()
        loss = criterion(data, out)
        loss.backward()
        optimizer.step()
        if verbose and it % 30 == 0:
            if gpu:
                ploss=loss.cpu.data[0]
            else:
                ploss=loss.data[0]
            print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(idx, run, it, ploss))

    #Evaluate
    mmd = 0
    for it in range(test_epochs):
        out = model()
        loss = criterion(data, out)
        if gpu:
            mmd += loss.cpu.data[0]
        else:
            mmd += loss.data[0]

    return mmd/test_epochs


def hill_climbing(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.nb_jobs) number of jobs
    :param kwargs: nb_runs=(SETTINGS.nb_runs) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.nb_jobs)
    nb_runs = kwargs.get("nb_runs", SETTINGS.nb_runs)
    loop = 0
    tested_configurations = [graph.get_dict_nw()]
    improvement = True
    result = []
    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
        data, graph, 0, run, **kwargs) for run in range(nb_runs))

    score_network = np.mean([i for i in result_pairs if np.isfinite(i)])
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
                result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                    data, test_graph, idx_pair, run, **kwargs) for run in range(nb_runs))

                score_network = np.mean([i for i in result_pairs if np.isfinite(i)])

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

    def orient_directed_graph(self, data, dag, alg='HC', **kwargs):
        """ Improve a directed acyclic graph using CGNN

        :param data: data
        :param dag: directed acyclic graph to optimize
        :param alg: type of algorithm
        :param log: Save logs of the execution
        :return: improved directed acyclic graph
        """
        alg_dic = {'HC': hill_climbing}
        return alg_dic[alg](dag, data, self.infer_graph, **kwargs)

    def orient_undirected_graph(self, data, umg, **kwargs):
        """ Orient the undirected graph using GNN and apply CGNN to improve the graph

        :param data: data
        :param umg: undirected acyclic graph
        :return: directed acyclic graph
        """

        warnings.warn("The pairwise GNN model is computed on each edge of the UMG "
                      "to initialize the model and start CGNN with a DAG")
        gnn = GNN(backend=self.backend, **kwargs)
        dag = gnn.orient_graph(data, umg, **kwargs)  # Pairwise method
        return self.orient_directed_graph(data, dag, **kwargs)
