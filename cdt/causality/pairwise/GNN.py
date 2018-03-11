"""
GNN : Generative Neural Networks for causal inference (pairwise)
Authors : Olivier Goudet & Diviyan Kalainathan
Ref:
Date : 10/05/2017
"""
import numpy as np
from ...utils.Loss import MMD_loss_th as MMD_th
from ...utils.Loss import TTestCriterion
from ...utils.Graph import DirectedGraph
from ...utils.Settings import SETTINGS, CGNN_SETTINGS
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
import torch as th
from torch.autograd import Variable
from .model import PairwiseModel
from pandas import DataFrame
from ...utils.Formats import reshape_data


class GNN_th(th.nn.Module):
    def __init__(self, batch_size, **kwargs):
        """
        Build the Torch graph
        :param batch_size: size of the batch going to be fed to the model
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim)
                       Number of units in the hidden layer
        :param kwargs: gpu=(SETTINGS.GPU), if GPU is used for computations
        :param kwargs: gpu_no=(0), GPU ID
        """
        super(GNN_th, self).__init__()
        h_layer_dim = kwargs.get('h_layer_dim', CGNN_SETTINGS.h_layer_dim)
        gpu = kwargs.get('gpu', SETTINGS.GPU)
        gpu_no = kwargs.get('gpu_no', 0)
        self.l1 = th.nn.Linear(2, h_layer_dim)
        self.l2 = th.nn.Linear(h_layer_dim, 1)
        self.noise = Variable(th.FloatTensor(
            batch_size, 1), requires_grad=False)
        if gpu:
            self.noise = self.noise.cuda(gpu_no)
        self.act = th.nn.ReLU()

    def forward(self, x):
        """
        Pass data through the net structure
        :param x: input data: shape (:,1)
        :type x: torch.Variable
        :return: output of the shallow net
        :rtype: torch.Variable

        """
        self.noise.normal_()
        y = self.act(self.l1(th.cat([x, self.noise], 1)))
        return self.l2(y)


def run_GNN_th(m, pair=0, run=0, **kwargs):
    """ Train and eval the GNN on a pair

    :param m: Matrix containing cause at m[:,0],
              and effect at m[:,1]
    :type m: numpy.ndarray
    :param pair: Number of the pair
    :param run: Number of the run
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used

    :param kwargs: test_epochs=(CGNN_SETTINGS.nb_epoch_test) test epochs
    :param kwargs: train_epochs=(CGNN_SETTINGS.nb_epoch_train) train epochs
    :param kwargs: learning_rate=(CGNN_SETTINGS.learning_rate)
    :return: Value of the evaluation after training
    :rtype: float
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    gpu_no = kwargs.get('gpu_no', 0)
    train_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.train_epochs)
    test_epochs = kwargs.get('test_epochs', CGNN_SETTINGS.test_epochs)
    learning_rate = kwargs.get('learning_rate', CGNN_SETTINGS.learning_rate)

    m = m.astype('float32')
    inputx = Variable(th.from_numpy(m[:, 0]))
    target = Variable(th.from_numpy(m[:, 1]))
    GNN = GNN_th(m.shape[0], **kwargs)

    if gpu:
        target = target.cuda(gpu_no)
        inputx = inputx.cuda(gpu_no)
        GNN = GNN.cuda(gpu_no)

    criterion = MMD_th(m.shape[0], cuda=gpu)

    optim = th.optim.Adam(GNN.parameters(), lr=learning_rate)
    running_loss = 0
    teloss = 0

    for i in range(train_epochs):
        optim.zero_grad()
        pred = GNN(inputx)
        loss = criterion(th.cat([inputx, target], 1),
                         th.cat([inputx, pred], 1))
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
        pred = GNN(inputx)
        loss = criterion(th.cat([inputx, target], 1),
                         th.cat([inputx, pred], 1))

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
            # fliplr is unsupported in Torch
            YX = run_GNN_th(m[:, [1, 0]], pair_idx, run, **kwargs)

    else:
        XY = run_GNN_th(m, pair_idx, run, **kwargs)
        YX = run_GNN_th(m, pair_idx, run, **kwargs)

    return [XY, YX]


# Test
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
                a, b, idx, run, **kwargs) for run in range(ttest_criterion.iter, ttest_criterion.iter + nb_runs))
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

            pred.append(self.predict_proba(
                a, b, idx, dim_variables_a=1, dim_variables_b=1, **kwargs))

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

            dim_variables_a = dim_variables_a[a]
            dim_variables_b = dim_variables_b[b]

            weight, p_val = self.predict_proba(
                data_a, data_b, idx, dim_variables_a=dim_variables_a, dim_variables_b=dim_variables_b, **kwargs)

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


if __name__ == "__main__":
    print("Testing GNN..")
    raise NotImplementedError
