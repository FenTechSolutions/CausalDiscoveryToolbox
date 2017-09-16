"""
CGNN_graph_model
Author : Olivier Goudet & Diviyan Kalainathan
Ref :
Date : 09/5/17
"""

import tensorflow as tf
import torch as th
from torch.autograd import Variable
from pandas import DataFrame
from sklearn.preprocessing import scale
import warnings
from joblib import Parallel, delayed
import sys
import numpy as np
from copy import deepcopy
from .model import GraphModel
from ..pairwise.GNN import GNN
from ...utils.Loss import MMD_loss_tf, MMD_loss_th, Fourier_MMD_Loss_tf, TTestCriterion
from ...utils.Loss import median_heursitic
from ...utils.Settings import SETTINGS
import pandas as pd


def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class CGNN_tf(object):
    def __init__(self, N, graph, run, idx, gamma, **kwargs):
        """ Build the tensorflow graph of the CGNN structure

        :param N: Number of points
        :param graph: Graph to be run
        :param run: number of the run (only for print)
        :param idx: number of the idx (only for print)
        :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
        :param kwargs: h_layer_dim=(SETTINGS.h_layer_dim) Number of units in the hidden layer
        :param kwargs: use_Fast_MMD=(SETTINGS.use_Fast_MMD) use fast MMD option
        :param kwargs: nb_vectors_approx_MMD=(SETTINGS.nb_vectors_approx_MMD) nb vectors
        """
        learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)
        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_layer_dim)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get('nb_vectors_approx_MMD', SETTINGS.nb_vectors_approx_MMD)

        self.run = run
        self.idx = idx
        list_nodes = graph.list_nodes()
        n_var = len(list_nodes)

        self.all_real_variables = tf.placeholder(tf.float32, shape=[None, n_var])

        generated_variables = {}
        theta_G = []

        while len(generated_variables) < n_var:
            # Need to generate all variables in the graph using its parents : possible because of the DAG structure
            for var in list_nodes:
                # Check if all parents are generated
                par = graph.parents(var)
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

        self.all_generated_variables = tf.concat(listvariablegraph, 1)

        if(use_Fast_MMD):
            self.G_dist_loss_xcausesy = Fourier_MMD_Loss_tf(self.all_real_variables, self.all_generated_variables,nb_vectors_approx_MMD, gamma)
        else:
            self.G_dist_loss_xcausesy = MMD_loss_tf(self.all_real_variables, self.all_generated_variables, gamma)

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
        :param kwargs: train_epochs=(SETTINGS.train_epochs) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', SETTINGS.train_epochs)
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
        :param kwargs: test_epochs=(SETTINGS.test_epochs) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', SETTINGS.test_epochs)
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

    def generate(self, data, **kwargs):

        generated_variables = self.sess.run([self.all_generated_variables], feed_dict={self.all_real_variables: data})

        tf.reset_default_graph()
        return np.array(generated_variables)[0, :, :]


def run_CGNN_tf(data, graph, idx=0, run=0, gamma=1, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param df_data: data corresponding to the graph
    :param graph: Graph to be run
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: nb_gpu=(SETTINGS.nb_gpu) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.gpu_offset) number of gpu offsets
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    nb_gpu = kwargs.get('nb_gpu', SETTINGS.NB_GPU)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.GPU_OFFSET)


    if (data.shape[0] > SETTINGS.max_nb_points):
        p = np.random.permutation(data.shape[0])
        data  = data[p[:int(SETTINGS.max_nb_points)],:]

    if gpu:
        with tf.device('/gpu:' + str(gpu_offset + run % nb_gpu)):
            model = CGNN_tf(data.shape[0], graph, run, idx, gamma, **kwargs)
            model.train(data, **kwargs)
            return model.evaluate(data, **kwargs)
    else:
        model = CGNN_tf(data.shape[0], graph, run, idx, gamma, **kwargs)
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
        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_layer_dim)

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
    :param kwargs: train_epochs=(SETTINGS.train_epochs) number of train epochs
    :param kwargs: test_epochs=(SETTINGS.test_epochs) number of test epochs
    :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
    :return: MMD loss value of the given structure after training

    """

    gpu = kwargs.get('gpu', SETTINGS.GPU)
    nb_gpu = kwargs.get('nb_gpu', SETTINGS.NB_GPU)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.GPU_OFFSET)
    train_epochs = kwargs.get('test_epochs', SETTINGS.train_epochs)
    test_epochs = kwargs.get('test_epochs', SETTINGS.test_epochs)
    learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)
    
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
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)
    nb_max_runs = kwargs.get("nb_max_runs", SETTINGS.NB_MAX_RUNS)
    id_run = kwargs.get("id_run", 0)
    ttest_threshold = kwargs.get("ttest_threshold", SETTINGS.ttest_threshold)
    loop = 0
    list_nodes = graph.list_nodes()
    data = data[list_nodes].as_matrix()

    median = median_heursitic(data)
    gamma = [0.1*median, 0.5*median, median, 2*median, 10*median]

    data = data.astype('float32')
     
    tested_configurations = [graph.dict_nw()]
    improvement = True
    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
        data, graph, 0, run, gamma, **kwargs) for run in range(nb_max_runs))
    best_structure_scores = [i for i in result_pairs if np.isfinite(i)]
    score_network = np.mean(best_structure_scores)
    globalscore = score_network

    print("Graph score : " + str(globalscore))

    while improvement:
        loop += 1
        improvement = False
        list_edges = graph.list_edges()
        for idx_pair in range(len(list_edges)):
            edge = list_edges[idx_pair]
            test_graph = deepcopy(graph)
            test_graph.reverse_edge(edge[0], edge[1])

            if (test_graph.is_cyclic()
                or test_graph.dict_nw() in tested_configurations):
                print('No Evaluation for {}'.format([edge]))
            else:
                print('Edge {} in evaluation :'.format(edge))
                tested_configurations.append(test_graph.dict_nw())
                ttest_criterion = TTestCriterion(max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)
                configuration_scores = []

                while ttest_criterion.loop(configuration_scores[:len(best_structure_scores)],
                                           best_structure_scores[:len(configuration_scores)]):
                    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                                    data, test_graph, idx_pair, run, gamma, **kwargs)
                                    for run in range(ttest_criterion.iter, ttest_criterion.iter+nb_runs))
                    configuration_scores.extend([i for i in result_pairs if np.isfinite(i)])
                score_network = np.mean(configuration_scores)

                print("Current score : {}".format(score_network))
                print("Best score : {}".format(globalscore))
                print("P-value : {}".format(ttest_criterion.p_value))
                if score_network < globalscore:
                    graph.reverse_edge(edge[0], edge[1])
                    improvement = True
                    if len(configuration_scores) < nb_max_runs:
                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                            data, test_graph, idx_pair, run, gamma, **kwargs)
                            for run in range(len(configuration_scores), nb_max_runs-len(configuration_scores)))
                        configuration_scores.extend([i for i in result_pairs if np.isfinite(i)])
                    print('Edge {} got reversed !'.format(edge))
                    globalscore = score_network
                    best_structure_scores = configuration_scores

                df_edge_result = pd.DataFrame(graph.list_edges(),
                                              columns=['Cause', 'Effect',
                                                       'Weight'])
                df_edge_result.to_csv('results/CGNN-HC' + str(id_run) + '-loop{}.csv'.format(loop), index=False)

    return graph


def exploratory_hill_climbing(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)

    nb_loops = 150
    exploration_factor = 10  # Average of number of edges to reverse at the beginning.
    assert exploration_factor < len(graph.list_edges())

    loop = 0
    tested_configurations = [graph.dict_nw()]
    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
        data, graph, 0, run, **kwargs) for run in range(nb_runs))

    score_network = np.mean([i for i in result_pairs if np.isfinite(i)])
    globalscore = score_network

    print("Graph score : " + str(globalscore))

    while loop < nb_loops:
        loop += 1
        list_edges = graph.list_edges()

        possible_solution=False
        while not possible_solution:
            test_graph = deepcopy(graph)
            selected_edges = np.random.choice(len(list_edges),
                                              max(int(exploration_factor * ((nb_loops-loop)/nb_loops)**2), 1))
            for edge in list_edges[selected_edges]:
                test_graph.reverse_edge()
            if not (test_graph.is_cyclic()
                    or test_graph.dict_nw() in tested_configurations):
                possible_solution = True

            print('Reversed Edges {} in evaluation :'.format(list_edges[selected_edges]))
            tested_configurations.append(test_graph.dict_nw())
            result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                data, test_graph, loop, run, **kwargs) for run in range(nb_runs))

            score_network = np.mean([i for i in result_pairs if np.isfinite(i)])

            print("Current score : " + str(score_network))
            print("Best score : " + str(globalscore))

            if score_network < globalscore:
                graph.reverse_edge(edge[0], edge[1])
                print('Edge {} got reversed !'.format(list_edges[selected_edges]))
                globalscore = score_network

    return graph


def tabu_search(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)
    raise ValueError('Not Yet Implemented')


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
        data = DataFrame(scale(data.as_matrix()), columns=data.columns)
        alg_dic = {'HC': hill_climbing, 'tabu': tabu_search, 'EHC': exploratory_hill_climbing}
        return alg_dic[alg](dag, data, self.infer_graph, **kwargs)

    def orient_undirected_graph(self, data, umg, **kwargs):
        """ Orient the undirected graph using GNN and apply CGNN to improve the graph

        :param data: data
        :param umg: undirected acyclic graph
        :return: directed acyclic graph
        """

        warnings.warn("The pairwise GNN model is computed on each edge of the UMG "
                      "to initialize the model and start CGNN with a DAG")
        data = DataFrame(scale(data.as_matrix()), columns=data.columns)
        gnn = GNN(backend=self.backend, **kwargs)
        dag = gnn.orient_graph(data, umg, **kwargs)  # Pairwise method
        return self.orient_directed_graph(data, dag, **kwargs)
