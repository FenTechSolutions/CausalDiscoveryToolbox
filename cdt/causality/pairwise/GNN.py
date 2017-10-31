"""
GNN : Generative Neural Networks for causal inference (pairwise)
Authors : Olivier Goudet & Diviyan Kalainathan
Ref:
Date : 10/05/2017
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
import tensorflow as tf
from .model import PairwiseModel
from pandas import DataFrame
from .GNN_th import th_run_instance
from ...utils.Formats import reshape_data
from ...utils.Loss import MMD_loss_tf as MMD_tf
from ...utils.Loss import Fourier_MMD_Loss_tf as Fourier_MMD_tf
from ...utils.Loss import TTestCriterion
from ...utils.Graph import DirectedGraph
from ...utils.Settings import SETTINGS, CGNN_SETTINGS

def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(CGNN_SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(CGNN_SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', CGNN_SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class GNN_tf(object):
    def __init__(self, N, run=0, pair=0, **kwargs):
        """ Build the tensorflow graph, the first column is set as the cause and the second as the effect

        :param N: Number of examples to generate
        :param run: for log purposes (optional)
        :param pair: for log purposes (optional)
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim) Number of units in the hidden layer
        :param kwargs: learning_rate=(CGNN_SETTINGS.learning_rate) learning rate of the optimizer
        :param kwargs: use_Fast_MMD=(CGNN_SETTINGS.use_Fast_MMD) use fast MMD option
        :param kwargs: nb_vectors_approx_MMD=(CGNN_SETTINGS.nb_vectors_approx_MMD) nb vectors
        """

        h_layer_dim = kwargs.get('h_layer_dim', CGNN_SETTINGS.h_layer_dim)
        learning_rate = kwargs.get(
            'learning_rate', CGNN_SETTINGS.learning_rate)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', CGNN_SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get(
            'nb_vectors_approx_MMD', CGNN_SETTINGS.nb_vectors_approx_MMD)

        self.run = run
        self.pair = pair
        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])

        W_in = tf.Variable(init([2, h_layer_dim], **kwargs))
        b_in = tf.Variable(init([h_layer_dim], **kwargs))
        W_out = tf.Variable(init([h_layer_dim, 1], **kwargs))
        b_out = tf.Variable(init([1], **kwargs))

        theta_G = [W_in, b_in,
                   W_out, b_out]

        e = tf.random_normal([N, 1], mean=0, stddev=1)

        hid = tf.nn.relu(tf.matmul(tf.concat([self.X, e], 1), W_in) + b_in)
        out_y = tf.matmul(hid, W_out) + b_out

        if (use_Fast_MMD):
            self.G_dist_loss_xcausesy = Fourier_MMD_tf(tf.concat([self.X, self.Y], 1), tf.concat([self.X, out_y], 1),
                                                       nb_vectors_approx_MMD)
        else:
            self.G_dist_loss_xcausesy = MMD_tf(
                tf.concat([self.X, self.Y], 1), tf.concat([self.X, out_y], 1))

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(learning_rate=learning_rate)
                                  .minimize(self.G_dist_loss_xcausesy, var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True, **kwargs):
        """ Train the GNN model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: train_epochs=(CGNN_SETTINGS.nb_epoch_train) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', CGNN_SETTINGS.train_epochs)

        for it in range(train_epochs):
            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.X: data[:, [0]], self.Y: data[:, [1]]}
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.pair, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True, **kwargs):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: test_epochs=(CGNN_SETTINGS.nb_epoch_test) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.test_epochs)
        avg_score = 0

        for it in range(test_epochs):
            score = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={
                                  self.X: data[:, [0]], self.Y: data[:, [1]]})

            avg_score += score[0]

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(
                        self.pair, self.run, it, score[0]))

        tf.reset_default_graph()

        return avg_score / test_epochs


def tf_evalcausalscore_pairwise(df, idx, run, **kwargs):
    GNN = GNN_tf(df.shape[0], run, idx, **kwargs)
    GNN.train(df, **kwargs)
    return GNN.evaluate(df, **kwargs)


def tf_run_instance(m, idx, run, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: gpu_list=(SETTINGS.GPU_LIST) List of CUDA_VISIBLE_DEVICES
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    gpu_list = kwargs.get('gpu_list', SETTINGS.GPU_LIST)

    if (m.shape[0] > CGNN_SETTINGS.max_nb_points):
        p = np.random.permutation(m.shape[0])
        m = m[p[:int(CGNN_SETTINGS.max_nb_points)], :]

    if gpu:
        with tf.device('/gpu:' + str(gpu_list[run % len(gpu_list)])):
            XY = tf_evalcausalscore_pairwise(m, idx, run, **kwargs)
        with tf.device('/gpu:' + str(gpu_list[run % len(gpu_list)])):
            YX = tf_evalcausalscore_pairwise(m[:, [1, 0]], idx, run, **kwargs)
            return [XY, YX]
    else:
        return [tf_evalcausalscore_pairwise(m, idx, run, **kwargs),
                tf_evalcausalscore_pairwise(np.fliplr(m), idx, run, **kwargs)]


class GNN(PairwiseModel):
    """
    Shallow Generative Neural networks, models the causal directions x->y and y->x with a 1-hidden layer neural network
    and a MMD loss. The causal direction is considered as the "best-fit" between the two directions
    """

    def __init__(self, backend="TensorFlow"):
        super(GNN, self).__init__()
        self.backend = backend

    def predict_proba(self, a, b, idx=0, **kwargs):

        backend_alg_dic = {"PyTorch": th_run_instance,
                           "TensorFlow": tf_run_instance}
        if len(np.array(a).shape) == 1:
            a = np.array(a).reshape((-1, 1))
            b = np.array(b).reshape((-1, 1))

        nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
        nb_runs = kwargs.get("nb_runs", CGNN_SETTINGS.NB_RUNS)
        nb_max_runs = kwargs.get("nb_max_runs", CGNN_SETTINGS.NB_MAX_RUNS)
        verbose = kwargs.get("verbose", SETTINGS.verbose)
        ttest_threshold = kwargs.get(
            "ttest_threshold", CGNN_SETTINGS.ttest_threshold)

        m = np.hstack((a, b))
        m = m.astype('float32')
        ttest_criterion = TTestCriterion(
            max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)

        AB = []
        BA = []

        while ttest_criterion.loop(AB, BA):
            result_pair = Parallel(n_jobs=nb_jobs)(delayed(backend_alg_dic[self.backend])(
                m, idx, run, **kwargs) for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))
            AB.extend([runpair[0] for runpair in result_pair])
            BA.extend([runpair[1] for runpair in result_pair])
        if verbose:
            print("P-value after {} runs : {}".format(ttest_criterion.iter,
                                                      ttest_criterion.p_value))
        score_AB = np.mean(AB)
        score_BA = np.mean(BA)

        return (score_BA - score_AB) / (score_BA + score_AB), ttest_criterion.p_value

    def predict_dataset(self, x, printout=None):
        """ Causal prediction of a pairwise dataset (x,y)

        :param x: Pairwise dataset
        :param printout: print regularly predictions
        :type x: cepc_df format
        :return: predictions probabilities
        :rtype: list
        """

        pred = []
        res = []
        for idx, row in x.iterrows():

            a = scale(row['A'].reshape((len(row['A']), 1)))
            b = scale(row['B'].reshape((len(row['B']), 1)))

            pred.append(self.predict_proba(a, b, idx))

            if printout is not None:
                res.append([row['SampleID'], pred[-1][0], pred[-1][1]])
                DataFrame(res, columns=['SampleID', 'Predictions', 'P-Value']).to_csv(
                    printout, index=False)
        return pred

    def orient_graph(self, df_data, type_variables, umg, deletion=False, printout=None):
        """ Orient an undirected graph using the pairwise method defined by the subclass
        Requirement : Name of the nodes in the graph correspond to name of the variables in df_data

        :param df_data: dataset
        :param umg: UndirectedGraph
        :param printout: print regularly predictions
        :return: Directed graph w/ weights
        :rtype: DirectedGraph
        """

        edges = umg.list_edges()
        list_nodes = umg.list_nodes()
        graph = DirectedGraph()
        res = []
        idx = 0

        for edge in edges:
            a, b, c = edge

            data, dim_variables = reshape_data(
                df_data, list_nodes, type_variables)

            weight, p_val = self.predict_proba(
                scale(df_data[a].as_matrix()), scale(df_data[b].as_matrix()), idx)

            if weight > 0:  # a causes b
                graph.add(a, b, weight)
            else:
                graph.add(b, a, abs(weight))
            if printout is not None:
                res.append([str(a),  str(b), weight, p_val])
                DataFrame(res, columns=['Var1', "Var2", 'Predictions', 'P_value']).to_csv(
                    printout, index=False)

            idx += 1
        if not deletion:
            graph.remove_cycles_without_deletion()
        else:
            graph.remove_cycles()

        return graph
