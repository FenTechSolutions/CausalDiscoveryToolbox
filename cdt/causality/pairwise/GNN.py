"""
GNN : Generative Neural Networks for causal inference (pairwise)
Authors : Olivier Goudet & Diviyan Kalainathan
Ref:
Date : 10/05/2017
"""
import tensorflow as tf
import numpy as np
from ...utils.Loss import MMD_loss_tf as MMD_tf
from ...utils.Loss import Fourier_MMD_Loss_tf as Fourier_MMD_tf
from ...utils.Loss import MMD_loss_th as MMD_th
from ...utils.Loss import TTestCriterion
from ...utils.Graph import DirectedGraph
from ...utils.Settings import SETTINGS
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
import torch as th
from torch.autograd import Variable
from .model import PairwiseModel
from pandas import DataFrame


def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class GNN_tf(object):
    def __init__(self, N, run=0, pair=0, **kwargs):
        """ Build the tensorflow graph, the first column is set as the cause and the second as the effect

        :param N: Number of examples to generate
        :param run: for log purposes (optional)
        :param pair: for log purposes (optional)
        :param kwargs: h_layer_dim=(SETTINGS.h_layer_dim) Number of units in the hidden layer
        :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
        :param kwargs: use_Fast_MMD=(SETTINGS.use_Fast_MMD) use fast MMD option
        :param kwargs: nb_vectors_approx_MMD=(SETTINGS.nb_vectors_approx_MMD) nb vectors
        """

        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_layer_dim)
        learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get('nb_vectors_approx_MMD', SETTINGS.nb_vectors_approx_MMD)

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
            self.G_dist_loss_xcausesy = MMD_tf(tf.concat([self.X, self.Y], 1), tf.concat([self.X, out_y], 1))

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
        :param kwargs: train_epochs=(SETTINGS.nb_epoch_train) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', SETTINGS.train_epochs)

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
        :param kwargs: test_epochs=(SETTINGS.nb_epoch_test) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', SETTINGS.test_epochs)
        avg_score = 0

        for it in range(test_epochs):
            score = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={self.X: data[:, [0]], self.Y: data[:, [1]]})

            avg_score += score[0]

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(self.pair, self.run, it, score[0]))

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
    :param kwargs: nb_gpu=(SETTINGS.NB_GPU) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.GPU_OFFSET) number of gpu offsets
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    nb_gpu = kwargs.get('nb_gpu', SETTINGS.NB_GPU)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.GPU_OFFSET)

    if (m.shape[0] > SETTINGS.max_nb_points):
        p = np.random.permutation(m.shape[0])
        m = m[p[:int(SETTINGS.max_nb_points)], :]

    run_i = run
    if gpu:
        with tf.device('/gpu:' + str(gpu_offset + run_i % nb_gpu)):
            XY = tf_evalcausalscore_pairwise(m, idx, run, **kwargs)
        with tf.device('/gpu:' + str(gpu_offset + run_i % nb_gpu)):
            YX = tf_evalcausalscore_pairwise(m[:, [1, 0]], idx, run, **kwargs)
            return [XY, YX]
    else:
        return [tf_evalcausalscore_pairwise(m, idx, run, **kwargs),
                tf_evalcausalscore_pairwise(np.fliplr(m), idx, run, **kwargs)]


class GNN_th(th.nn.Module):
    def __init__(self, **kwargs):
        """
        Build the Torch graph
        :param kwargs: h_layer_dim=(SETTINGS.h_layer_dim) Number of units in the hidden layer
        """
        super(GNN_th, self).__init__()
        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_layer_dim)
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
    :param kwargs: train_epochs=(SETTINGS.nb_epoch_train) number of train epochs
    :param kwargs: test_epochs=(SETTINGS.nb_epoch_test) number of test epochs
    :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
    :return: Value of the evaluation after training
    :rtype: float
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    train_epochs = kwargs.get('test_epochs', SETTINGS.train_epochs)
    test_epochs = kwargs.get('test_epochs', SETTINGS.test_epochs)
    learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)

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
            YX = run_GNN_th(m[:, [1, 0]], pair_idx, run, **kwargs)  # fliplr is unsupported in Torch

    else:
        XY = run_GNN_th(m, pair_idx, run, **kwargs)
        YX = run_GNN_th(m, pair_idx, run, **kwargs)

    return [XY, YX]


class GNN(PairwiseModel):
    """
    Shallow Generative Neural networks, models the causal directions x->y and y->x with a 1-hidden layer neural network
    and a MMD loss. The causal direction is considered as the "best-fit" between the two directions
    """

    def __init__(self, backend="PyTorch"):
        super(GNN, self).__init__()
        self.backend = backend

    def predict_proba(self, a, b, idx=0, **kwargs):

        backend_alg_dic = {"PyTorch": th_run_instance, "TensorFlow": tf_run_instance}
        if len(np.array(a).shape) == 1:
            a = np.array(a).reshape((-1, 1))
            b = np.array(b).reshape((-1, 1))

        nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
        nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)
        nb_max_runs = kwargs.get("nb_max_runs", SETTINGS.NB_MAX_RUNS)
        verbose= kwargs.get("verbose", SETTINGS.verbose)
        ttest_threshold = kwargs.get("ttest_threshold", SETTINGS.ttest_threshold)

        m = np.hstack((a, b))
        m = m.astype('float32')
        ttest_criterion = TTestCriterion(max_iter=nb_max_runs, runs_per_iter=nb_runs, threshold=ttest_threshold)

        AB = []
        BA = []

        while ttest_criterion.loop(AB, BA):
            result_pair = Parallel(n_jobs=nb_jobs)(delayed(backend_alg_dic[self.backend])(
                m, idx, run, **kwargs) for run in range(ttest_criterion.iter, ttest_criterion.iter+nb_runs))
            AB.extend([runpair[0] for runpair in result_pair])
            BA.extend([runpair[1] for runpair in result_pair])
        if verbose:
            print("P-value after {} runs : {}".format(ttest_criterion.iter, ttest_criterion.p_value))
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

    def orient_graph(self, df_data, umg, deletion=False, printout=None):
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

        for edge in edges:
            a, b, c = edge
            weight, p_val = self.predict_proba(scale(df_data[a].as_matrix()), scale(df_data[b].as_matrix()), idx)

            if weight > 0:  # a causes b
                graph.add(a, b, weight)
            else:
                graph.add(b, a, abs(weight))
            if printout is not None:
                res.append([str(a) + '-' + str(b), weight, p_val])
                DataFrame(res, columns=['SampleID', 'Predictions', 'P_value']).to_csv(
                    printout, index=False)

            idx += 1
        if not deletion:
            graph.remove_cycles_without_deletion()
        else:
            graph.remove_cycles()

        return graph
