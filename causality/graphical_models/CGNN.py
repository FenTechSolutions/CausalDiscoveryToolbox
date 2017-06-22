"""
CGNN_graph_model
Author : Olivier Goudet & Diviyan Kalainathan
Ref :
Date : 09/5/17
"""

import tensorflow as tf
import torch as th
import warnings
import os
import pandas as pd
from joblib import Parallel, delayed
import sys
from copy import deepcopy
from .model import GraphModel
from ..pairwise_models.SGNN import SGNN
from ...utils.loss import MMD_loss
from ...utils.Graph import *
from ...utils.SETTINGS import CGNN_SETTINGS as SETTINGS


def init(size):
    return tf.random_normal(shape=size, stddev=SETTINGS.init_weights)


class CGNN_graph_tf(object):
    def __init__(self, N, graph, list_nodes, run, pair, learning_rate=SETTINGS.learning_rate):
        """
        Build the tensorflow graph,
        For a given structure
        """
        self.run = run
        self.pair = pair
        n_var = len(list_nodes)

        self.all_real_variables = tf.placeholder(tf.float32, shape=[None, n_var])

        generated_variables = {}
        theta_G = []

        while len(generated_variables) < n_var:
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

        self.G_dist_loss_xcausesy = MMD_loss(self.all_real_variables, all_generated_variables)

        # var_list = theta_G
        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True):
        for it in range(SETTINGS.nb_epoch_train):

            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_real_variables: data}
            )

            if verbose:
                if (it % 100 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.pair, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True):

        sumMMD_tr = 0

        for it in range(SETTINGS.nb_epoch_test):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={
                self.all_real_variables: data})

            sumMMD_tr += MMD_tr[0]

            if verbose:
                if (it % 100 == 0):
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'
                          .format(self.pair, self.run, it, MMD_tr[0]))

        tf.reset_default_graph()

        return sumMMD_tr / SETTINGS.nb_epoch_test


def run_graph_tf(df_data, graph, idx, run):
    list_nodes = graph.get_list_nodes()
    df_data = df_data[list_nodes].as_matrix()
    data = df_data.astype('float32')

    if SETTINGS.GPU:
        with tf.device('/gpu:' + str(SETTINGS.gpu_offset + run % SETTINGS.num_gpu)):
            model = CGNN_graph_tf(df_data.shape[0], graph, list_nodes, run, idx)
            model.train(data)
            return model.evaluate(data)
    else:
        model = CGNN_graph_tf(df_data.shape[0], graph, list_nodes, run, idx)
        model.train(data)
        return model.evaluate(data)

name_algo = "HC_CGNN_"

dataset_name = ((sys.argv[1].split('.'))[0]).split('/')[-1]
print(dataset_name)


def save_log(pair_scores, filename, evalgraph=False):
    if not evalgraph:
        r_df = pd.DataFrame(pair_scores, columns=[
            'SampleID', 'edge_no', 'score_XY', 'score_YX', 'run'])
    else:
        r_df = pd.DataFrame(pair_scores, columns=[
            'SampleID', 'change_edge_no', 'score', 'run'])

    if not os.path.exists('results/'):
        os.makedirs('results/')

    r_df.to_csv(filename, index=False)


def unpack_results(result_pairs, pair_scores, node1, node2, i, loop=0, evalgraph=False, log=False):
    """  Process the results given by the multiprocessing loop

    :param result_pairs:
    :param pair_scores:
    :param node1:
    :param node2:
    :param i:
    :param loop:
    :param evalgraph:
    :return:
    """
    run_no = 0

    if not evalgraph:
        sum_score_XY = 0
        sum_score_YX = 0

        for result_pair in result_pairs:

            run_no += 1

            score_XY = result_pair[0]
            score_YX = result_pair[1]

            if np.isfinite(score_XY) and np.isfinite(score_YX):
                sum_score_XY += score_XY
                sum_score_YX += score_YX

            else:
                warnings.warn('NaN value', RuntimeWarning)

            if log:
                pair_scores.append(
                    [str(node1) + '-' + str(node2), i, score_XY, score_YX, run_no])
                save_log(pair_scores, 'results/results' + '_pw_' +
                         dataset_name + name_algo + '.csv')

        return sum_score_XY, sum_score_YX

    else:
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

            if log:
                pair_scores.append(
                    [str(node1) + '-' + str(node2), i, score_graph, run_no])
                save_log(pair_scores, 'results/results' + '_' + dataset_name +
                         name_algo + str(loop) + '.csv', evalgraph=True)

        return sum_score_graph / denom


def infer_graph_tf(data, dag, log=False):

    result = []
    loop = 0
    improvement = True

    result_pairs = Parallel(n_jobs=SETTINGS.nb_jobs)(delayed(run_graph_tf)(
        data, dag, 0, run) for run in range(SETTINGS.nb_run))
    tested_configurations = [dag.get_dict_nw()]
    score_network = unpack_results(
        result_pairs, result, "", "", 0, loop, evalgraph=True, log=log)
    globalscore = score_network

    print("Graph score : " + str(globalscore))

    while improvement:
        loop += 1
        improvement = False
        list_edges = dag.get_list_edges()
        for idx_pair in range(len(list_edges)):
            edge = list_edges[idx_pair]
            test_dag = deepcopy(dag)
            test_dag.reverse_edge(edge[0], edge[1])

            if (test_dag.get_dict_nw().is_cyclic()
                or test_dag.get_dict_nw() in tested_configurations):
                print('No Evaluation for {}'.format([edge]))
            else:
                print('Edge {} in evaluation :'.format(edge))
                tested_configurations.append(test_dag.get_dict_nw())
                result_pairs = Parallel(n_jobs=SETTINGS.nb_jobs)(delayed(run_graph_tf)(
                    data, test_dag, idx_pair, run) for run in range(SETTINGS.nb_run))

                score_network = unpack_results(
                    result_pairs, result, edge[0], edge[1], idx_pair, loop, evalgraph=True, log=log)
                score_network = score_network

                print("Current score : " + str(score_network))
                print("Best score : " + str(globalscore))

                if score_network < globalscore:
                    # , globalscore - score_network)
                    dag.reverse_edge(edge[0], edge[1])
                    improvement = True
                    print('Edge {} got reversed !'.format(edge))
                    globalscore = score_network

                if log:
                    df_edge_result = pd.DataFrame(dag.tolist(),
                                                  columns=['Cause', 'Effect', 'Weight'])
                    df_edge_result.to_csv('results/' + name_algo + dataset_name +
                                          '-loop{}.csv'.format(loop), index=False)
    return dag


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

    def create_graph_from_data(self, data):
        print("The CGNN model is not able (yet) to model the graph directly from raw data")
        raise ValueError

    def orient_directed_graph(self, data, dag, log=False):
        """

        :param data:
        :param dag:
        :param log: Save logs of the execution
        :return:
        """
        if self.backend == 'TensorFlow':
            return infer_graph_tf(data, dag, log=False)
        elif self.backend == 'PyTorch':
            pass
        else:
            print('No backend known as {}'.format(self.backend))
            raise ValueError

    def orient_undirected_graph(self, data, umg):
        """

        :param data:
        :param umg:
        :return:
        """

        warnings.warn("The pairwise GNN model is computed once on the UMG to initialize the model and start with a DAG")
        Gnn = SGNN(backend=self.backend)
        dag = Gnn.orient_graph(data, umg)  # Pairwise method
        return self.orient_directed_graph(data,dag)
