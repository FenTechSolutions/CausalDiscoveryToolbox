"""
CGNN_graph_model
Author : Olivier Goudet & Diviyan Kalainathan
Ref :
Date : 09/5/17
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.preprocessing import scale

from ...utils.Loss import MMD_loss_tf, Fourier_MMD_Loss_tf
from ...utils.Settings import SETTINGS, CGNN_SETTINGS
from .model import GraphModel
from ...utils.Formats import reshape_data
from ...utils.Graph import Block
from ...utils.Graph import DirectedGraph

from pymprog import *

import random

def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', CGNN_SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class CGNN_decomposable_tf(object):
    def __init__(self, N,  n_target, n_parents, n_all_other_variables, run, idx, **kwargs):
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
        learning_rate = kwargs.get('learning_rate', CGNN_SETTINGS.learning_rate)
        h_layer_dim = kwargs.get('h_layer_dim', CGNN_SETTINGS.h_layer_dim)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', CGNN_SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get('nb_vectors_approx_MMD', CGNN_SETTINGS.nb_vectors_approx_MMD)

        self.run = run
        self.idx = idx

        self.all_other_variables = tf.placeholder(tf.float32, shape=[None, n_all_other_variables])
        self.target_variables = tf.placeholder(tf.float32, shape=[None, n_target])
        self.parents_variables = tf.placeholder(tf.float32, shape=[None, n_parents])

        theta_G = []

        W_in = tf.Variable(init([n_parents + 1, h_layer_dim], **kwargs))
        b_in = tf.Variable(init([h_layer_dim], **kwargs))
        W_out = tf.Variable(init([h_layer_dim, 1], **kwargs))
        b_out = tf.Variable(init([1], **kwargs))

        theta_G.extend([W_in, b_in, W_out, b_out])

        input_v = tf.concat([self.parents_variables, tf.random_normal([N, 1], mean=0, stddev=1)], 1)
        out_v = tf.nn.relu(tf.matmul(input_v, W_in) + b_in)
        out_v = tf.matmul(out_v, W_out) + b_out

        self.all_real_variables = tf.concat([self.all_other_variables, self.target_variables], 1)
        self.all_generated_variables = tf.concat([self.all_other_variables, out_v] , 1)

        if(use_Fast_MMD):
            self.G_dist_loss_xcausesy = Fourier_MMD_Loss_tf(self.all_real_variables, self.all_generated_variables, nb_vectors_approx_MMD)
        else:
            self.G_dist_loss_xcausesy = MMD_loss_tf(self.all_real_variables, self.all_generated_variables)

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data_target, data_all_other_variables, data_parents, verbose=True, **kwargs):
        """ Train the initialized model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: train_epochs=(SETTINGS.train_epochs) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', CGNN_SETTINGS.train_epochs)
        for it in range(train_epochs):

            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_other_variables: data_all_other_variables, self.target_variables: data_target, self.parents_variables: data_parents }
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.idx, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data_target, data_all_other_variables, data_parents, verbose=True, **kwargs):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: test_epochs=(SETTINGS.test_epochs) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.test_epochs)
        sumMMD_tr = 0

        for it in range(test_epochs):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={self.all_other_variables: data_all_other_variables, self.target_variables: data_target, self.parents_variables: data_parents })

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




class CGNN_all_blocks(object):

    def __init__(self, df_data, umg, run, **kwargs):


        learning_rate = kwargs.get('learning_rate', CGNN_SETTINGS.learning_rate)
        h_layer_dim = kwargs.get('h_layer_dim', CGNN_SETTINGS.h_layer_dim)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', CGNN_SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get('nb_vectors_approx_MMD', CGNN_SETTINGS.nb_vectors_approx_MMD)

        self.run = run

        list_nodes = umg.list_nodes()

        data = df_data.as_matrix()
        data = data.reshape(data.shape[0], data.shape[1])
        N = data.shape[0]

        self.all_variables = tf.placeholder(tf.float32, shape=[None, df_data.shape[1]])

        G_dist_loss = 0
        self.dict_all_W_in = {}

        for target in list_nodes:

            num_target = int(target[1:])

            list_neighbours = umg.neighbors(target)

            list_num_neighbours = []
            for node in list_neighbours:
                list_num_neighbours.append(int(node[1:]))

            list_all_other_variables = list(df_data.columns.values)
            list_all_other_variables.remove(target)

            list_num_all_other_variables = []
            for node in list_all_other_variables:
                list_num_all_other_variables.append(int(node[1:]))

            all_neighbour_variables = tf.transpose(tf.gather(tf.transpose(self.all_variables) , np.array(list_num_neighbours)))

            target_variable = tf.transpose(tf.gather(tf.transpose(self.all_variables), num_target))
            target_variable = tf.reshape(target_variable, [N,1])

            all_other_variables = tf.transpose(tf.gather(tf.transpose(self.all_variables) , np.array(list_num_all_other_variables)))

            list_W_in = []
            dict_target_W_in = {}
            for node in list_neighbours:
                W_in = tf.Variable(init([1, CGNN_SETTINGS.h_layer_dim]))
                dict_target_W_in[node] = tf.reduce_sum(tf.abs(W_in))
                list_W_in.append(W_in)

            self.dict_all_W_in[target] = dict_target_W_in

            W_noise = tf.Variable(init([1, CGNN_SETTINGS.h_layer_dim]))
            W_input = tf.concat([tf.concat(list_W_in,0),W_noise ], 0)

            b_in = tf.Variable(init([CGNN_SETTINGS.h_layer_dim]))
            W_out = tf.Variable(init([CGNN_SETTINGS.h_layer_dim, 1]))
            b_out = tf.Variable(init([1]))

            input = tf.concat([all_neighbour_variables, tf.random_normal([N, 1], mean=0, stddev=1)], 1)
            output = tf.nn.relu(tf.matmul(input, W_input) + b_in)
            output = tf.matmul(output, W_out) + b_out
            output = tf.reshape(output, [N, 1])

            if (use_Fast_MMD):
                G_dist_loss += Fourier_MMD_Loss_tf(tf.concat([tf.concat(all_other_variables, 1),target_variable],1), tf.concat([tf.concat(all_other_variables, 1),output],1), nb_vectors_approx_MMD)
            else:
                G_dist_loss += MMD_loss_tf(tf.concat([tf.concat(all_other_variables, 1),target_variable],1), tf.concat([tf.concat(all_other_variables, 1),output],1))

        asymmetry_constraint = 0

        for edge in umg.list_edges():
            asymmetry_constraint += CGNN_SETTINGS.asymmetry_param * self.dict_all_W_in[edge[0]][edge[1]] * self.dict_all_W_in[edge[1]][edge[0]]

        self.G_global_loss = G_dist_loss + asymmetry_constraint

        self.G_solver = (tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.G_global_loss))

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
        train_epochs = kwargs.get('train_epochs', CGNN_SETTINGS.train_epochs)
        for it in range(train_epochs):

            _, G_global_loss_curr = self.sess.run([self.G_solver, self.G_global_loss],feed_dict={self.all_variables: data})

            if verbose:
                if it % 100 == 0:
                    print('Run:{}, Iter:{}, score:{}'.format(self.run, it, G_global_loss_curr))


    def evaluate(self, data, umg, verbose=True, **kwargs):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: test_epochs=(SETTINGS.test_epochs) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.test_epochs)

        nb_nodes = len(umg.list_nodes())
        matrix_results = np.zeros((nb_nodes,nb_nodes))

        for it in range(test_epochs):

            _, G_global_loss_curr, dict_all_W_in_curr = self.sess.run([self.G_solver, self.G_global_loss, self.dict_all_W_in],feed_dict={self.all_variables: data})

            if verbose:
                if it % 100 == 0:
                    print('Run:{}, Iter:{}, score:{}'.format(self.run, it, G_global_loss_curr))


            for edge in umg.list_edges():

                print(edge[0] + " -> " + edge[1] + " : " + str(dict_all_W_in_curr[edge[0]][edge[1]]))
                print(edge[1] + " -> " + edge[0] + " : " + str(dict_all_W_in_curr[edge[1]][edge[0]]))

                matrix_results[int(edge[0][1:]), int(edge[1][1:])] += dict_all_W_in_curr[edge[0]][edge[1]]
                matrix_results[int(edge[1][1:]), int(edge[0][1:])] += dict_all_W_in_curr[edge[1]][edge[0]]


        tf.reset_default_graph()

        return matrix_results




def run_CGNN_decomposable_tf(df_data, type_variables, graph, list_parents, target, idx=0, run=0,  **kwargs):
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
    gpu_list = kwargs.get('gpu_list', SETTINGS.GPU_LIST)

    print("target : " + str(target))
    print(type_variables)

    data_target,_ = reshape_data(df_data, list([target]), type_variables)

    list_all_other_variables = graph.list_nodes()
    list_all_other_variables.remove(target)

    data_all_other_variables,_ = reshape_data(df_data, list_all_other_variables, type_variables)

    if(len(list_parents)>0):
        data_parents,_ = reshape_data(df_data, list(list_parents), type_variables)
    else:
        data_parents = df_data[list(list_parents)].as_matrix()


    print("target " + str(target))
    print("list_parents")
    print(list_parents)
    print("list_all_other_variables")
    print(list_all_other_variables)


    if gpu:
        with tf.device('/gpu:' + str(gpu_list[run % len(gpu_list)])):
            model = CGNN_decomposable_tf(df_data.shape[0], data_target.shape[1], data_parents.shape[1], data_all_other_variables.shape[1], run, idx, **kwargs)
            model.train(data_target, data_all_other_variables, data_parents, **kwargs)
            score_node = model.evaluate(data_target, data_all_other_variables, data_parents, **kwargs)
    else:
        model = CGNN_decomposable_tf(df_data.shape[0], data_target.shape[1], data_parents.shape[1], data_all_other_variables.shape[1],  run, idx, **kwargs)
        model.train(data_target, data_all_other_variables, data_parents, **kwargs)
        score_node = model.evaluate(data_target, data_all_other_variables, data_parents, **kwargs)

    print("score node : " + str(score_node))

    return score_node


def hill_climbing(graph, data, type_variables, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", CGNN_SETTINGS.NB_RUNS)
    loop = 0
    tested_configurations = [graph.dict_nw()]
    improvement = True
    result = []
    list_eval_nodes = graph.list_nodes()

    list_tested_block = []

    cpt = 0
    for target in list_eval_nodes:
        
        list_parents = graph.parents(target)

        result_node = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, type_variables, graph, list_parents, target, cpt, run, **kwargs) for run in range(nb_runs))
        cpt += 1
        score_node = np.mean([i for i in result_node if np.isfinite(i)])
        graph.score_node[target] = score_node
        
        block = Block(target, list_parents)
        block.score = score_node

        list_tested_block.append(block)
        
 
    globalscore = graph.total_score()

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

                for target in [edge[0],edge[1]]:

                    list_parents = test_graph.parents(target)
                     
                    found = False
                    score = 0
                    for block in list_tested_block:
                        if(target == block.node and list_parents == block.parents):
                            found = True
                            score = block.score

                    if(found):
                        score_block = score
                    else:
                        result_block = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, type_variables, test_graph, test_graph.parents(target), target, idx_pair, run, **kwargs) for run in range(nb_runs))
                        score_block = np.mean([i for i in result_block if np.isfinite(i)])
        
                        block = Block(target, list_parents) 
                        block.score = score_node
                        list_tested_block.append(block)

                    test_graph.score_node[target] = score_block

                score_network = test_graph.total_score()

                print("Current score : " + str(score_network))
                print("Best score : " + str(globalscore))

                if score_network < globalscore:
                    graph = deepcopy(test_graph)
                    improvement = True
                    print('Edge {} got reversed !'.format(edge))
                    globalscore = score_network

    return graph



def eval_all_possible_blocks(data, type_variables, umg, run_cgnn_function, **kwargs):

    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", CGNN_SETTINGS.NB_RUNS)

    max_parents_block = kwargs.get("max_parents_block", CGNN_SETTINGS.max_parents_block)

    list_nodes = umg.list_nodes()
    list_block = []

    idx_block = 0

    for node in list_nodes:

        list_neighbours = umg.parents(node)

        for L in range(0, len(list_neighbours) + 1):
            for set_parents in itertools.combinations(list_neighbours, L):

                if(len(set_parents) <= max_parents_block):
                    block = Block(node,set_parents)
                    block.idx = idx_block
                    idx_block += 1
                    list_block.append(block)

    print("nb bloks to evaluate : " + str(idx_block))


    for block in list_block:
        result_block = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, type_variables, umg, block.parents, block.node, block.idx, run, **kwargs) for run in range(nb_runs))
        block.score = np.mean([i for i in result_block if np.isfinite(i)])
        #block.score = random.random()

    file = open("save_block.csv", "w")
    for block in list_block:
        file.write(str(block.idx) + ";" + str(block.node) + ";" + str(block.parents) + ";" + str(block.score) + "\n")

    return list_block



def solve_ilp_problem(list_blocks, umg):


    p = model("dag")

    list_edges = umg.list_edges(return_weights=False)
    list_nodes = umg.list_nodes()

    block_var = p.var('block', len(list_blocks), bool)
    edge_var = p.var('edge', len(list_edges), bool)

    nb_blocks = len(list_blocks)

    p.minimize(sum(list_blocks[t].score * block_var[t] for t in range(len(list_blocks))))

    for node in list_nodes:
        sum(block_var[j] for j in range(nb_blocks) if list_blocks[j].node == node) == 1

    # block_edge constraints:
    # if a block is selected, all the corresponding edges must be selected
    for num_block, block in enumerate(list_blocks):
        print(num_block)
        if (len(block.parents) > 0):
            list_corresponding_edges = []
            
            for parent in block.parents:
                print("parent " + str(parent))

                edge_block = [parent, block.node]

                for num_edge, edge in enumerate(list_edges):

                    if (edge_block == edge):

                        block_var[num_block] - edge_var[num_edge] <= 0.0
                        list_corresponding_edges.append(num_edge)

    # block constraints:
    # if an edge is selected, one corresponding block must be selected
    for num_edge, edge in enumerate(list_edges):
        list_corresponding_blocks = []
        for num_block, block in enumerate(list_blocks):
            for parent in block.parents:
                if ([parent, block.node] == edge):
                    list_corresponding_blocks.append(num_block)
        if (len(list_corresponding_blocks) > 0):
            sum(block_var[t] for t in list_corresponding_blocks) - edge_var[num_edge] >= 0

    for num_edge, edge in enumerate(list_edges):
        for i in range(num_edge + 1, len(list_edges)):
            new_edge = list_edges[i]
            if (edge[0] == new_edge[1] and edge[1] == new_edge[0]):
                edge_var[num_edge] + edge_var[i] == 1

    list_cycles = umg.cycles()
    print("nb cycles " + str(len(list_cycles)))

    for cycle in list_cycles:
       if(len(cycle) > 3):
           list_edges_cycle = []
           for n in range(len(cycle)-1):
               node0 = cycle[n]
               node1 = cycle[n+1]
               for num_edge, edge in enumerate(list_edges):
                   if (edge[0] == node0 and edge[1] == node1):
                       list_edges_cycle.append(num_edge)

           sum(edge_var[t] for t in list_edges_cycle) <= len(cycle)-2

    p.solver('intopt')

    p.solve(float)  # solve as LP only.
    print("simplex done: %r" % p.status())
    p.solve(int)  # solve the IP problem
    print("score " + str(p.vobj()))

    # print(" ")
    # print("solution blocks")
    #
    # for num_block, block in enumerate(block_var):
    #     print("cause " + str(list_blocks[num_block].parents) + " effect " + str(list_blocks[num_block].node) )
    #     print(block.primal)
    #     print(list_blocks[num_block].score)

    solution_graph = DirectedGraph()

    for num_edge, edge in enumerate(edge_var):
        if(edge.primal == 1):

            for num_block, block in enumerate(block_var):

                if(list_blocks[num_block].node == list_edges[num_edge][1] and block.primal == 1):

                    solution_graph.add(list_edges[num_edge][0],list_edges[num_edge][1], list_blocks[num_block].score)


    print(solution_graph.is_cyclic())

    return solution_graph

def get_random_graph(umg):

    edges = umg.list_edges_without_duplicate()
    graph = DirectedGraph()
    for edge in edges:
        a, b = edge
        if random.random() < 0.5:
            graph.add(a, b, 1)
        else:
            graph.add(b, a, abs(1))
    graph.remove_cycle_without_deletion()
    return graph


def run_CGNN_all_blocks(df_data, umg, run,**kwargs):

    gpu = kwargs.get('gpu', SETTINGS.GPU)
    gpu_list = kwargs.get('gpu_list', SETTINGS.GPU_LIST)

    df_data = pd.DataFrame(scale(df_data), columns=df_data.columns)

    if gpu:
        with tf.device('/gpu:' + str(gpu_list[run % len(gpu_list)])):
            model = CGNN_all_blocks(df_data, umg, run, **kwargs)
            model.train(df_data.as_matrix(), **kwargs)
            matrix_result = model.evaluate(df_data.as_matrix(),umg, **kwargs)
    else:
        model = CGNN_all_blocks(df_data, umg, run, **kwargs)
        model.train(df_data.as_matrix(), **kwargs)
        matrix_result = model.evaluate(df_data.as_matrix(), umg, **kwargs)

    return matrix_result




def embedded_method(data, umg,**kwargs):

    list_nodes = list(data.columns.values)
    matrix_results = np.zeros((len(list_nodes), len(list_nodes)))

    result_matrix = Parallel(n_jobs=SETTINGS.NB_JOBS)(delayed(run_CGNN_all_blocks)(data, umg, run,**kwargs ) for run in range(CGNN_SETTINGS.NB_RUNS))

    for i in range(len(result_matrix)):
        matrix_results += result_matrix[i]

    matrix_results = matrix_results / CGNN_SETTINGS.NB_RUNS

    dag = DirectedGraph()

    for edge in umg.list_edges():

        score_edge = matrix_results[int(edge[0][1:]),int(edge[1][1:])] - matrix_results[int(edge[1][1:]),int(edge[0][1:])]
        if(score_edge > 0):
            dag.add(edge[0],edge[1], score_edge)
        else:
            dag.add(edge[1], edge[0], -score_edge)

    return dag



def load_block(df_block):

    list_block = []
    idx_block = 0
    for i in range(df_block.shape[0]):

        node = str(df_block["Node"].loc[i])
        parent_str = df_block["Parents"].loc[i]
        parent_str = parent_str.replace("(", "")
        parent_str = parent_str.replace(")", "")
        parent_str = parent_str.replace("'", "")
        parent_str = parent_str.replace(" ", "")
        set_parents = set(str.split(parent_str,","))

        print(set_parents)
 
        block = Block(node,set_parents, idx_block)
        block.score = float(df_block["Score"].loc[i])
        idx_block += 1
        list_block.append(block)

    return list_block


class CGNN_decomposable(GraphModel):
    """
    CGNN Model ; Using generative models, generate the whole causal graph and improve causal
    direction predictions in the graph.
    """

    def __init__(self, backend='TensorFlow'):
        """ Initialize the CGNN Model.

        :param backend: Choose the backend to use, either 'PyTorch' or 'TensorFlow'
        """
        super(CGNN_decomposable, self).__init__()
        self.backend = backend

        if self.backend == 'TensorFlow':
            self.infer_graph = run_CGNN_decomposable_tf
        else:
            print('No backend known as {}'.format(self.backend))
            raise ValueError

    def create_graph_from_data(self, data):
        print("The CGNN model is not able (yet?) to model the graph directly from raw data")
        raise ValueError

    def orient_directed_graph(self, data, type_variables, dag, alg='HC', **kwargs):
        """ Improve a directed acyclic graph using CGNN

        :param data: data
        :param dag: directed acyclic graph to optimize
        :param alg: type of algorithm
        :param log: Save logs of the execution
        :return: improved directed acyclic graph
        """

        alg_dic = {'HC': hill_climbing}
        return alg_dic[alg](dag, data, type_variables, self.infer_graph, **kwargs)

    def orient_undirected_graph(self, data, type_variables, umg, mode, name_graph = "", saved_blocks = None, **kwargs):
        """ Orient the undirected graph using GNN and apply CGNN to improve the graph

        :param data: data
        :param umg: undirected acyclic graph
        :return: directed acyclic graph
        """
        if(mode == 0):
            if saved_blocks is not None:
                result_blocks = load_block(saved_blocks)
                return solve_ilp_problem(result_blocks, umg)
            else:
                result_blocks = eval_all_possible_blocks(data, type_variables, umg, self.infer_graph, **kwargs)
                file = open(name_graph + "_save_block.csv", "w")
                for block in result_blocks:
                    file.write(str(block.idx) + ";" + str(block.node) + ";" + str(block.parents) + ";" + str(block.score) + "\n")
                return solve_ilp_problem(result_blocks, umg)

        elif(mode == 1):

            return self.orient_directed_graph(data,type_variables, get_random_graph(umg) , **kwargs)

        elif (mode == 2):

            return embedded_method(data, umg, **kwargs)
