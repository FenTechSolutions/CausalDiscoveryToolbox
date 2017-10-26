"""
CGNN_graph_model
Author : Olivier Goudet & Diviyan Kalainathan
Ref :
Date : 09/5/17
"""

import tensorflow as tf
from pandas import DataFrame
from sklearn.preprocessing import scale
import warnings
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from copy import deepcopy
from .model import GraphModel
from ..pairwise.GNN import GNN
from ...utils.Loss import MMD_loss_tf, Fourier_MMD_Loss_tf, TTestCriterion
from ...utils.Settings import SETTINGS, CGNN_SETTINGS
from ...utils.Formats import reshape_data


def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(CGNN_SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(CGNN_SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', CGNN_SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class CGNN_tf(object):
    def __init__(self, N, dim_variables, graph, run, idx, with_confounders=False, **kwargs):
        """ Build the tensorflow graph of the CGNN structure

        :param N: Number of points
        :param graph: Graph to be run
        :param run: number of the run (only for print)
        :param idx: number of the idx (only for print)
        :param kwargs: learning_rate=(CGNN_SETTINGS.learning_rate) learning rate of the optimizer
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim) Number of units in the hidden layer
        :param kwargs: use_Fast_MMD=(CGNN_SETTINGS.use_Fast_MMD) use fast MMD option, Fourier Approx.
        :param kwargs: nb_vectors_approx_MMD=(CGNN_SETTINGS.nb_vectors_approx_MMD) nb vectors
        """
        learning_rate = kwargs.get(
            'learning_rate', CGNN_SETTINGS.learning_rate)
        h_layer_dim = kwargs.get('h_layer_dim', CGNN_SETTINGS.h_layer_dim)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', CGNN_SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get(
            'nb_vectors_approx_MMD', CGNN_SETTINGS.nb_vectors_approx_MMD)

        self.run = run
        self.idx = idx
        list_nodes = graph.list_nodes()
        n_var = len(list_nodes)

        tot_dim = 0
        for i in list_nodes:
            tot_dim += dim_variables[i]

        self.all_real_variables = tf.placeholder(
            tf.float32, shape=[None, tot_dim])

        generated_variables = {}
        theta_G = []

        if(with_confounders):
            list_edges = graph.skeleton.list_edges()
            confounder_variables = {}
            for edge in list_edges:
                noise_variable = tf.random_normal([N, 1], mean=0, stddev=1)
                confounder_variables[edge[0], edge[1]] = noise_variable
                confounder_variables[edge[1], edge[0]] = noise_variable

        while len(generated_variables) < n_var:
            # Need to generate all variables in the graph using its parents : possible because of the DAG structure
            for var in list_nodes:
                # Check if all parents are generated
                par = graph.parents(var)

                if (var not in generated_variables and set(par).issubset(generated_variables)):

                    n_parents = 0
                    for i in par:
                        n_parents += dim_variables[i]

                    if(with_confounders):
                        neighboorhood = graph.skeleton.neighbors(var)
                        W_in = tf.Variable(
                            init([n_parents + len(neighboorhood) + 1, h_layer_dim], **kwargs))
                    else:
                        W_in = tf.Variable(
                            init([n_parents + 1, h_layer_dim], **kwargs))

                    b_in = tf.Variable(init([h_layer_dim], **kwargs))
                    W_out = tf.Variable(
                        init([h_layer_dim, dim_variables[var]], **kwargs))
                    b_out = tf.Variable(init([dim_variables[var]], **kwargs))

                    input_v = [generated_variables[i] for i in par]
                    input_v.append(tf.random_normal([N, 1], mean=0, stddev=1))

                    if(with_confounders):
                        for i in neighboorhood:
                            input_v.append(confounder_variables[i, var])

                    input_v = tf.concat(input_v, 1)

                    out_v = tf.nn.relu(tf.matmul(input_v, W_in) + b_in)

                    if(dim_variables[var] == 1):
                        out_v = tf.matmul(out_v, W_out) + b_out
                    else:
                        out_v = tf.nn.softmax(tf.matmul(out_v, W_out) + b_out)

                    generated_variables[var] = out_v
                    theta_G.extend([W_in, b_in, W_out, b_out])

        listvariablegraph = []
        for var in list_nodes:
            listvariablegraph.append(generated_variables[var])

        self.all_generated_variables = tf.concat(listvariablegraph, 1)

        if(use_Fast_MMD):
            self.G_dist_loss_xcausesy = Fourier_MMD_Loss_tf(
                self.all_real_variables, self.all_generated_variables, nb_vectors_approx_MMD)
        else:
            self.G_dist_loss_xcausesy = MMD_loss_tf(
                self.all_real_variables, self.all_generated_variables)

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
        :param kwargs: train_epochs=(CGNN_SETTINGS.train_epochs) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', CGNN_SETTINGS.train_epochs)
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
        :param kwargs: test_epochs=(CGNN_SETTINGS.test_epochs) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.test_epochs)
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

        generated_variables = self.sess.run([self.all_generated_variables], feed_dict={
                                            self.all_real_variables: data})

        tf.reset_default_graph()
        return np.array(generated_variables)[0, :, :]


def run_CGNN_tf(data, type_variables, graph, idx=0, run=0, with_confounders=False, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param df_data: data corresponding to the graph
    :param graph: Graph to be run
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: gpu_list=(SETTINGS.GPU_LIST) List of CUDA_VISIBLE_DEVICES
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    gpu_list = kwargs.get('gpu_list', SETTINGS.GPU_LIST)

    data, dim_variables = reshape_data(df_data, list_nodes, type_variables)
    data = data.astype('float32')
    if (data.shape[0] > CGNN_SETTINGS.max_nb_points):

        p = np.random.permutation(data.shape[0])
        data = data[p[:int(CGNN_SETTINGS.max_nb_points)], :]

    if gpu:
        with tf.device('/gpu:' + str(gpu_list[run % len(gpu_list)])):
            model = CGNN_tf(data.shape[0], dim_variables,
                            graph, run, idx, with_confounders, **kwargs)
            model.train(data, **kwargs)
            return model.evaluate(data, **kwargs)
    else:
        model = CGNN_tf(data.shape[0], dim_variables,
                        graph, run, idx, with_confounders, **kwargs)
        model.train(data, **kwargs)
        return model.evaluate(data, **kwargs)


def hill_climbing(graph, data, type_variables, run_cgnn_function, with_confounders=False, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(CGNN_SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", CGNN_SETTINGS.NB_RUNS)
    nb_max_runs = kwargs.get("nb_max_runs", CGNN_SETTINGS.NB_MAX_RUNS)
    id_run = kwargs.get("id_run", 0)
    ttest_threshold = kwargs.get(
        "ttest_threshold", CGNN_SETTINGS.ttest_threshold)
    printout = kwargs.get("printout", None)
    loop = 0

    tested_configurations = [graph.dict_nw()]
    improvement = True
    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
        data, type_variables, graph, 0, run, with_confounders, **kwargs) for run in range(nb_max_runs))
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
                ttest_criterion = TTestCriterion(
                    max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)
                configuration_scores = []

                while ttest_criterion.loop(configuration_scores[:len(best_structure_scores)],
                                           best_structure_scores[:len(configuration_scores)]):
                    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                        data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                        for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))
                    configuration_scores.extend(
                        [i for i in result_pairs if np.isfinite(i)])

                score_network = np.mean(configuration_scores)

                print("Current score : {}".format(score_network))
                print("Best score : {}".format(globalscore))
                print("P-value : {}".format(ttest_criterion.p_value))

                if score_network < globalscore:
                    graph.reverse_edge(edge[0], edge[1])
                    improvement = True
                    if len(configuration_scores) < nb_max_runs:
                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                            data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                            for run in range(len(configuration_scores), nb_max_runs - len(configuration_scores)))
                        configuration_scores.extend(
                            [i for i in result_pairs if np.isfinite(i)])

                    print('Edge {} got reversed !'.format(edge))
                    globalscore = np.mean(configuration_scores)
                    best_structure_scores = configuration_scores

                if printout is not None:
                    df_edge_result = pd.DataFrame(graph.list_edges(),
                                                  columns=['Cause', 'Effect',
                                                           'Weight'])
                    df_edge_result.to_csv(
                        printout + str(id_run) + '-loop{}.csv'.format(loop), index=False)

    return graph


def hill_climbing_with_removal(graph, data, type_variables, run_cgnn_function, with_confounders=False, **kwargs):
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
    ttest_threshold = kwargs.get("ttest_threshold", SETTINGS.ttest_threshold)

    loop = 0
    tested_configurations = [graph.dict_nw()]
    improvement = True

    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
        data, type_variables, graph, 0, run, with_confounders, **kwargs) for run in range(nb_max_runs))

    complexity_score = SETTINGS.complexity_graph_param * \
        len(graph.list_edges())
    best_structure_scores = [(i + complexity_score)
                             for i in result_pairs if np.isfinite(i)]

    score_network = np.mean(best_structure_scores)

    globalscore = score_network

    while improvement:

        loop += 1
        improvement = False
        list_edges_to_evaluate = graph.skeleton.list_edges()

        for idx_pair in range(0, len(list_edges_to_evaluate)):

            edge = list_edges_to_evaluate[idx_pair]

            print(edge)
            print(graph.list_edges(return_weights=False))
            # If edge already oriented in the graph
            if([edge[0], edge[1]] in graph.list_edges(return_weights=False) or [edge[1], edge[0]] in graph.list_edges(return_weights=False)):

                if([edge[0], edge[1]] in graph.list_edges(return_weights=False)):
                    node1 = edge[0]
                    node2 = edge[1]
                else:
                    node2 = edge[0]
                    node1 = edge[1]

                # Test reverse edge
                test_graph = deepcopy(graph)
                test_graph.reverse_edge(node1, node2)

                if (test_graph.is_cyclic() or test_graph.dict_nw() in tested_configurations):
                    print("No evaluation for edge " +
                          str(node1) + " -> " + str(node2))
                else:
                    print("Reverse Edge " + str(node1) +
                          " -> " + str(node2) + " in evaluation")
                    tested_configurations.append(test_graph.dict_nw())

                    ttest_criterion = TTestCriterion(
                        max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)
                    configuration_scores = []

                    while ttest_criterion.loop(configuration_scores[:len(best_structure_scores)], best_structure_scores[:len(configuration_scores)]):

                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                                                                for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))

                        complexity_score = SETTINGS.complexity_graph_param * \
                            len(test_graph.list_edges())
                        configuration_scores.extend(
                            [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                    score_network = np.mean(configuration_scores)

                    print("Current score : " + str(score_network))
                    print("Best score : " + str(globalscore))

                    if score_network < globalscore:

                        graph.reverse_edge(node1, node2)
                        improvement = True
                        print("Edge " + str(node1) + "->" +
                              str(node2) + " got reversed !")

                        if len(configuration_scores) < nb_max_runs:
                            result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                                data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                                for run in range(len(configuration_scores),
                                                 nb_max_runs - len(
                                    configuration_scores)))

                            complexity_score = SETTINGS.complexity_graph_param * \
                                len(test_graph.list_edges())
                            configuration_scores.extend(
                                [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                        globalscore = np.mean(configuration_scores)
                        best_structure_scores = configuration_scores

                        node = node1
                        node1 = node2
                        node2 = node

                # Test suppression
                test_graph = deepcopy(graph)
                test_graph.remove_edge(node1, node2)

                if (test_graph.dict_nw() in tested_configurations):
                    print("Removing already evaluated for edge " +
                          str(node1) + " -> " + str(node2))
                else:
                    print("Removing edge " + str(node1) +
                          " -> " + str(node2) + " in evaluation")

                    tested_configurations.append(test_graph.dict_nw())

                    ttest_criterion = TTestCriterion(
                        max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)
                    configuration_scores = []

                    while ttest_criterion.loop(configuration_scores[:len(best_structure_scores)], best_structure_scores[:len(configuration_scores)]):

                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                                                                for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))

                        complexity_score = SETTINGS.complexity_graph_param * \
                            len(test_graph.list_edges())
                        configuration_scores.extend(
                            [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                    score_network = np.mean(configuration_scores)

                    print("Current score : " + str(score_network))
                    print("Best score : " + str(globalscore))

                    if score_network < globalscore:
                        graph.remove_edge(node1, node2)
                        improvement = True
                        print("Edge " + str(node1) + " -> " +
                              str(node2) + " got removed !")

                        if len(configuration_scores) < nb_max_runs:
                            result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                                data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                                for run in range(len(configuration_scores),
                                                 nb_max_runs - len(
                                    configuration_scores)))

                            complexity_score = SETTINGS.complexity_graph_param * \
                                len(test_graph.list_edges())
                            configuration_scores.extend(
                                [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                        globalscore = np.mean(configuration_scores)
                        best_structure_scores = configuration_scores

                    else:
                        # We keep the edge and its score is set to (score_network - globalscore)
                        print("Edge " + str(node1) + " -> " + str(node2) +
                              " not removed. Score edge : " + str(score_network - globalscore))
                        graph.add(node1, node2, score_network - globalscore)

            # Eval if a suppressed edge need to be restored
            else:

                node1 = edge[0]
                node2 = edge[1]

                # Test add edge sens node1 -> node2
                test_graph = deepcopy(graph)
                test_graph.add(node1, node2)

                score_network_add_edge_node1_node2 = 9999

                if (test_graph.is_cyclic() or test_graph.dict_nw() in tested_configurations):
                    print("No addition possible for " +
                          str(node1) + " -> " + str(node2))
                else:
                    print("Addition of edge " + str(node1) +
                          " -> " + str(node2) + " in evaluation :")
                    tested_configurations.append(test_graph.dict_nw())

                    ttest_criterion = TTestCriterion(
                        max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)
                    configuration_scores_add_edge_node1_node2 = []

                    while ttest_criterion.loop(configuration_scores_add_edge_node1_node2[:len(best_structure_scores)], best_structure_scores[:len(configuration_scores_add_edge_node1_node2)]):

                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                                                                for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))

                        complexity_score = SETTINGS.complexity_graph_param * \
                            len(test_graph.list_edges())
                        configuration_scores_add_edge_node1_node2.extend(
                            [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                    score_network_add_edge_node1_node2 = np.mean(
                        configuration_scores_add_edge_node1_node2)

                    print("score network add edge " + str(node1) + " -> " +
                          str(node2) + " : " + str(score_network_add_edge_node1_node2))

                # Test add edge sens node2 -> node1
                test_graph = deepcopy(graph)
                test_graph.add(node2, node1)

                score_network_add_edge_node2_node1 = 9999

                if (test_graph.is_cyclic()
                        or test_graph.dict_nw() in tested_configurations):
                    print("No addition possible for edge " +
                          str(node2) + " -> " + str(node1))
                else:
                    print("Addition of edge " + str(node2) +
                          " -> " + str(node1) + " in evaluation :")
                    tested_configurations.append(test_graph.dict_nw())

                    ttest_criterion = TTestCriterion(
                        max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)
                    configuration_scores_add_edge_node2_node1 = []

                    while ttest_criterion.loop(configuration_scores_add_edge_node2_node1[:len(best_structure_scores)], best_structure_scores[:len(configuration_scores_add_edge_node2_node1)]):

                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                                                                for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))

                        complexity_score = SETTINGS.complexity_graph_param * \
                            len(test_graph.list_edges())
                        configuration_scores_add_edge_node2_node1.extend(
                            [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                    score_network_add_edge_node2_node1 = np.mean(
                        configuration_scores_add_edge_node2_node1)

                    print("score network add edge " + str(node2) + " -> " +
                          str(node1) + " : " + str(score_network_add_edge_node2_node1))

                print("Best score : " + str(globalscore))

                if score_network_add_edge_node1_node2 < globalscore and score_network_add_edge_node1_node2 < score_network_add_edge_node2_node1:

                    score_edge = globalscore - score_network_add_edge_node1_node2
                    graph.add(node1, node2, score_edge)
                    improvement = True
                    print("Edge " + str(node1) + " -> " + str(node2) +
                          " is added with score : " + str(score_edge) + " !")

                    if len(configuration_scores_add_edge_node1_node2) < nb_max_runs:
                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                            data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                            for run in range(len(configuration_scores_add_edge_node1_node2),
                                             nb_max_runs - len(configuration_scores_add_edge_node1_node2)))

                        complexity_score = SETTINGS.complexity_graph_param * \
                            len(test_graph.list_edges())
                        configuration_scores_add_edge_node1_node2.extend(
                            [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                    globalscore = np.mean(score_network_add_edge_node1_node2)
                    best_structure_scores = configuration_scores_add_edge_node1_node2

                elif score_network_add_edge_node2_node1 < globalscore and score_network_add_edge_node2_node1 < score_network_add_edge_node1_node2:
                    score_edge = globalscore - score_network_add_edge_node2_node1
                    graph.add(node2, node1, score_edge)
                    improvement = True
                    print("Edge " + str(node2) + " -> " + str(node1) +
                          " is added with score : " + str(score_edge) + " !")

                    if len(configuration_scores_add_edge_node2_node1) < nb_max_runs:
                        result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                            data, type_variables, test_graph, idx_pair, run, with_confounders, **kwargs)
                            for run in range(len(configuration_scores_add_edge_node2_node1),
                                             nb_max_runs - len(configuration_scores_add_edge_node2_node1)))

                        complexity_score = SETTINGS.complexity_graph_param * \
                            len(test_graph.list_edges())
                        configuration_scores_add_edge_node2_node1.extend(
                            [(i + complexity_score) for i in result_pairs if np.isfinite(i)])

                    globalscore = np.mean(
                        configuration_scores_add_edge_node2_node1)
                    best_structure_scores = configuration_scores_add_edge_node2_node1

                else:
                    print("Edge not added " + str(node1) + " | " + str(node2))

    return graph


def exploratory_hill_climbing(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(CGNN_SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", CGNN_SETTINGS.NB_RUNS)

    nb_loops = 150
    # Average of number of edges to reverse at the beginning.
    exploration_factor = 10
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

        possible_solution = False
        while not possible_solution:
            test_graph = deepcopy(graph)
            selected_edges = np.random.choice(len(list_edges),
                                              max(int(exploration_factor * ((nb_loops - loop) / nb_loops)**2), 1))
            for edge in list_edges[selected_edges]:
                test_graph.reverse_edge()
            if not (test_graph.is_cyclic()
                    or test_graph.dict_nw() in tested_configurations):
                possible_solution = True

            print('Reversed Edges {} in evaluation :'.format(
                list_edges[selected_edges]))
            tested_configurations.append(test_graph.dict_nw())
            result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                data, test_graph, loop, run, **kwargs) for run in range(nb_runs))

            score_network = np.mean(
                [i for i in result_pairs if np.isfinite(i)])

            print("Current score : " + str(score_network))
            print("Best score : " + str(globalscore))

            if score_network < globalscore:
                graph.reverse_edge(edge[0], edge[1])
                print('Edge {} got reversed !'.format(
                    list_edges[selected_edges]))
                globalscore = score_network

    return graph


def tabu_search(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(CGNN_SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", CGNN_SETTINGS.NB_RUNS)
    raise ValueError('Not Yet Implemented')


class CGNN(GraphModel):
    """
    CGNN Model ; Using generative models, generate the whole causal graph and improve causal
    direction predictions in the graph.
    """

    def __init__(self, backend='TensorFlow'):
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

        raise ValueError(
            "The CGNN model is not able to model the graph directly from raw data")

    def orient_directed_graph(self, data, dag, alg='HC', with_confounders=False, **kwargs):
        """ Improve a directed acyclic graph using CGNN

        :param data: data
        :param dag: directed acyclic graph to optimize
        :param alg: type of algorithm
        :param log: Save logs of the execution
        :return: improved directed acyclic graph
        """

        alg_dic = {'HC': hill_climbing, 'HCr': hill_climbing_with_removal,
                   'tabu': tabu_search, 'EHC': exploratory_hill_climbing}
        return alg_dic[alg](dag, data, type_variables, self.infer_graph, with_confounders, **kwargs)

    def orient_undirected_graph(self, data, type_variables, umg, with_confounders=False, **kwargs):
        """ Orient the undirected graph using GNN and apply CGNN to improve the graph

        :param data: data
        :param umg: undirected acyclic graph
        :return: directed acyclic graph
        """

        warnings.warn("The pairwise GNN model is computed on each edge of the UMG "
                      "to initialize the model and start CGNN with a DAG")

        gnn = GNN(backend=self.backend, **kwargs)
        dag = gnn.orient_graph(data, type_variables, umg,
                               **kwargs)  # Pairwise method

        return self.orient_directed_graph(data, type_variables, dag, with_confounders, **kwargs)
