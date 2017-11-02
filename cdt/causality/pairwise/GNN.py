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
from ...utils.Formats import reshape_data
from ...utils.Loss import MMD_loss_tf as MMD_tf
from ...utils.Loss import Fourier_MMD_Loss_tf as Fourier_MMD_tf
from ...utils.Loss import TTestCriterion
from ...utils.Graph import DirectedGraph
from ...utils.Settings import SETTINGS, CGNN_SETTINGS

if SETTINGS.torch is not None:
    from .GNN_th import th_run_instance
else:
    th_run_instance = None


def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(CGNN_SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(CGNN_SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', CGNN_SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class GNN_tf(object):
    def __init__(self, N, dim_variables_a, dim_variables_b, run=0, pair=0, **kwargs):
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
        self.X = tf.placeholder(tf.float32, shape=[None, dim_variables_a])
        self.Y = tf.placeholder(tf.float32, shape=[None, dim_variables_b])

        if(dim_variables_a == 1 and dim_variables_b == 1):

            W_in = tf.Variable(
                init([dim_variables_a + 1, h_layer_dim], **kwargs))
            b_in = tf.Variable(init([h_layer_dim], **kwargs))
            W_out = tf.Variable(init([h_layer_dim, dim_variables_b], **kwargs))
            b_out = tf.Variable(init([dim_variables_b], **kwargs))

            theta_G = [W_in, b_in, W_out, b_out]

            e = tf.random_normal([N, 1], mean=0, stddev=1)

            hid = tf.nn.relu(tf.matmul(tf.concat([self.X, e], 1), W_in) + b_in)

            if(dim_variables_b == 1):
                out_y = tf.matmul(hid_y, W_out) + b_out
            else:
                out_y = tf.nn.softmax(tf.matmul(out_y, W_out) + b_out)

            if (use_Fast_MMD):
                self.G_dist_loss_xcausesy = Fourier_MMD_tf(tf.concat(
                    [self.X, self.Y], 1), tf.concat([self.X, out_y], 1), nb_vectors_approx_MMD)
            else:
                self.G_dist_loss_xcausesy = MMD_tf(
                    tf.concat([self.X, self.Y], 1), tf.concat([self.X, out_y], 1))

        else:
            print("OK")

            Wx_in = tf.Variable(init([1, h_layer_dim], **kwargs))
            bx_in = tf.Variable(init([h_layer_dim], **kwargs))
            Wx_out = tf.Variable(
                init([h_layer_dim, dim_variables_a], **kwargs))
            bx_out = tf.Variable(init([dim_variables_a], **kwargs))

            Wy_in = tf.Variable(
                init([dim_variables_a + 1, h_layer_dim], **kwargs))
            by_in = tf.Variable(init([h_layer_dim], **kwargs))
            Wy_out = tf.Variable(
                init([h_layer_dim, dim_variables_b], **kwargs))
            by_out = tf.Variable(init([dim_variables_b], **kwargs))

            theta_G = [Wx_in, bx_in, Wx_out, bx_out,
                       Wy_in, by_in, Wy_out, by_out]

            ex = tf.random_normal([N, 1], mean=0, stddev=1)
            ey = tf.random_normal([N, 1], mean=0, stddev=1)

            hid_x = tf.nn.relu(tf.matmul(ex, Wx_in) + bx_in)

            if(dim_variables_a == 1):
                out_x = tf.matmul(hid_x, Wx_out) + bx_out
            else:
                out_x = tf.nn.softmax(tf.matmul(hid_x, Wx_out) + bx_out)

            hid_y = tf.nn.relu(
                tf.matmul(tf.concat([out_x, ey], 1), Wy_in) + by_in)

            if(dim_variables_b == 1):
                out_y = tf.matmul(hid_y, Wy_out) + by_out
            else:
                out_y = tf.nn.softmax(tf.matmul(hid_y, Wy_out) + by_out)

            if (use_Fast_MMD):
                self.G_dist_loss_xcausesy = Fourier_MMD_tf(tf.concat(
                    [self.X, self.Y], 1), tf.concat([out_x, out_y], 1), nb_vectors_approx_MMD)
            else:
                self.G_dist_loss_xcausesy = MMD_tf(
                    tf.concat([self.X, self.Y], 1), tf.concat([out_x, out_y], 1))

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy, var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, a, b, verbose=True, **kwargs):
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
                feed_dict={self.X: a, self.Y: b}
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.pair, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, a, b, verbose=True, **kwargs):
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
                                  self.X: a, self.Y: b})

            avg_score += score[0]

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(
                        self.pair, self.run, it, score[0]))

        tf.reset_default_graph()

        return avg_score / test_epochs


def tf_evalcausalscore_pairwise(a, b, dim_variables_a, dim_variables_b, idx, run, **kwargs):
    GNN = GNN_tf(a.shape[0], dim_variables_a,
                 dim_variables_b, run, idx, **kwargs)
    GNN.train(a, b, **kwargs)
    return GNN.evaluate(a, b, **kwargs)


def tf_run_instance(a, b, dim_variables_a, dim_variables_b, idx, run, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: nb_gpu=(SETTINGS.NB_GPU) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.GPU_OFFSET) number of gpu offsets
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    gpu_list = kwargs.get('nb_gpu', SETTINGS.GPU_LIST)

    if (a.shape[0] > CGNN_SETTINGS.max_nb_points):
        p = np.random.permutation(a.shape[0])
        a = a[p[:int(CGNN_SETTINGS.max_nb_points)], :]
        b = b[p[:int(CGNN_SETTINGS.max_nb_points)], :]

    if gpu:
        with tf.device('/gpu:' + str(gpu_list[run % len(gpu_list)])):
            XY = tf_evalcausalscore_pairwise(
                a, b, dim_variables_a, dim_variables_b, idx, run, **kwargs)
        with tf.device('/gpu:' + str(gpu_list[run % len(gpu_list)])):
            YX = tf_evalcausalscore_pairwise(
                b, a, dim_variables_b, dim_variables_a, idx, run, **kwargs)
            return [XY, YX]
    else:
        return [tf_evalcausalscore_pairwise(a, b, dim_variables_a, dim_variables_b, idx, run, **kwargs),
                tf_evalcausalscore_pairwise(b, a, dim_variables_b, dim_variables_a, idx, run, **kwargs)]


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

        dim_variables_a = kwargs.get("dim_variables_a", 1)
        dim_variables_b = kwargs.get("dim_variables_b", 1)
        nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
        nb_runs = kwargs.get("nb_runs", CGNN_SETTINGS.NB_RUNS)
        nb_max_runs = kwargs.get("nb_max_runs", CGNN_SETTINGS.NB_MAX_RUNS)
        verbose = kwargs.get("verbose", SETTINGS.verbose)
        ttest_threshold = kwargs.get(
            "ttest_threshold", CGNN_SETTINGS.ttest_threshold)

        a = a.astype('float32')
        b = b.astype('float32')

        ttest_criterion = TTestCriterion(
            max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)

        AB = []
        BA = []

        while ttest_criterion.loop(AB, BA):
            result_pair = Parallel(n_jobs=nb_jobs)(delayed(backend_alg_dic[self.backend])(
                a, b, dim_variables_a, dim_variables_b, idx, run, **kwargs) for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))
            AB.extend([runpair[0] for runpair in result_pair])
            BA.extend([runpair[1] for runpair in result_pair])

        if verbose:
            print("P-value after {} runs : {}".format(ttest_criterion.iter,
                                                      ttest_criterion.p_value))

        score_AB = np.mean(AB)
        score_BA = np.mean(BA)

        return (score_BA - score_AB) / (score_BA + score_AB), ttest_criterion.p_value

    def predict_dataset(self, x, **kwargs):
        """ Causal prediction of a pairwise dataset (x,y)

        :param x: Pairwise dataset
        :param printout: print regularly predictions
        :type x: cepc_df format
        :return: predictions probabilities
        :rtype: list
        """

        printout = kwargs.get("printout", None)

        pred = []
        res = []
        for idx, row in x.iterrows():

            a = scale(row['A'].reshape((len(row['A']), 1)))
            b = scale(row['B'].reshape((len(row['B']), 1)))

            pred.append(self.predict_proba(a, b, 1, 1, idx))

            if printout is not None:
                res.append([row['SampleID'], pred[-1][0], pred[-1][1]])
                DataFrame(res, columns=['SampleID', 'Predictions', 'P-Value']).to_csv(
                    printout, index=False)
        return pred

    def orient_graph(self, df_data, umg, **kwargs):
        """ Orient an undirected graph using the pairwise method defined by the subclass
        Requirement : Name of the nodes in the graph correspond to name of the variables in df_data

        :param df_data: dataset
        :param umg: UndirectedGraph
        :param printout: print regularly predictions
        :return: Directed graph w/ weights
        :rtype: DirectedGraph
        """

        edges = umg.list_edges()
        graph = DirectedGraph()
        res = []
        idx = 0

        deletion = kwargs.get("deletion", False)
        printout = kwargs.get("printout", None)
        type_variables = kwargs.get("type_variables", None)
        if type_variables is None:
            type_variables = {}
            for node in df_data.columns:
                type_variables[node] = "Numerical"

        for edge in edges:
            a, b, c = edge

            data_a, dim_variables_a = reshape_data(
                df_data, [a], type_variables)
            data_b, dim_variables_b = reshape_data(
                df_data, [b], type_variables)

            weight, p_val = self.predict_proba(
                data_a, data_b, dim_variables_a[a], dim_variables_b[b], idx)

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
