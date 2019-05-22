"""GNN : Generative Neural Networks for causal inference (pairwise).

Authors : Olivier Goudet & Diviyan Kalainathan
Ref: Causal Generative Neural Networks (https://arxiv.org/abs/1711.08936)
Date : 10/05/2017

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
import numpy as np
import torch as th
import networkx as nx
from tqdm import trange
from pandas import DataFrame
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from .model import PairwiseModel
from ...utils.loss import MMDloss, TTestCriterion
from ...utils.Settings import SETTINGS
from ...utils.parallel import parallel_run
from ...utils.io import MetaDataset, SimpleDataset


class GNN_model(th.nn.Module):
    """Torch model for the GNN structure."""

    def __init__(self, batch_size, nh=20, lr=0.01, train_epochs=1000, test_epochs=1000, idx=0,
            verbose=None, **kwargs):
        """Build the Torch graph.

        :param batch_size: size of the batch going to be fed to the model
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim)
                       Number of units in the hidden layer
        :param device: device on with the algorithm is going to be run on.
        """
        super(GNN_model, self).__init__()
        self.l1 = th.nn.Linear(2, nh)
        self.l2 = th.nn.Linear(nh, 1)
        self.register_buffer('noise', th.FloatTensor(batch_size, 1))
        self.act = th.nn.ReLU()
        self.criterion = MMDloss(batch_size)
        self.layers = th.nn.Sequential(self.l1, self.act, self.l2)

    def forward(self, x):
        """Pass data through the net structure.

        :param x: input data: shape (:,1)
        :type x: torch.Variable
        :return: output of the shallow net
        :rtype: torch.Variable

        """
        self.noise.normal_()
        return self.layers(th.cat([x, self.noise], 1))

    def run(self, x, y):
        """Run the GNN on a pair x,y of FloatTensor data."""
        verbose = SETTINGS.get_default(verbose=verbose)
        optim = th.optim.Adam(self.parameters(), lr=lr)
        running_loss = 0
        teloss = 0
        pbar = trange(train_epochs + test_epochs, disable=not verbose)
        for i in pbar:
            optim.zero_grad()
            pred = self.forward(x)
            loss = self.criterion(pred, y)
            running_loss += loss.item()

            if i < train_epochs:
                loss.backward()
                optim.step()
            else:
                teloss += running_loss

            # print statistics
            if not i % 300:
                pbar.set_postfix(idx=idx, score=running_loss/300)
                running_loss = 0.0

        return teloss / test_epochs

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


def GNN_instance(x, idx=0, device=None, nh=20, **kwargs):
    """Run an instance of GNN, testing causal direction.

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param pair_idx: print purposes
    :param run: numner of the run (for GPU dispatch)
    :param device: device on with the algorithm is going to be run on.
    :return:
    """
    device = SETTINGS.get_default(device=device)
    inputx = th.FloatTensor(xy[:, [0]]).to(device)
    target = th.FloatTensor(xy[:, [1]]).to(device)
    GNNXY = GNN_model(x.shape[0], nh=nh).to(device)
    GNNYX = GNN_model(x.shape[0], nh=nh).to(device)
    GNNXY.reset_parameters()
    GNNYX.reset_parameters()
    XY = GNNXY.run(inputx, target, **kwargs)
    YX = GNNYX.run(target, inputx, **kwargs)

    return [XY, YX]


class GNN(PairwiseModel):
    """Shallow Generative Neural networks.

    Models the causal directions x->y and y->x with a 1-hidden layer neural
    network and a MMD loss. The causal direction is considered as the "best-fit"
    between the two causal directions.

    Args:
        nh (int): number of hidden units in the neural network
        lr (float): learning rate of the optimizer
        nruns (int): number of runs to execute per batch
           (before testing for significance with t-test).
        njobs (int): number of runs to execute in parallel.
           (defaults to ``cdt.SETTINGS.NJOBS``)
        gpus (bool): Number of available gpus
           (defaults to ``cdt.SETTINGS.GPU``)
        idx (int): (optional) index of the pair, for printing purposes
        verbose (bool): verbosity (defaults to ``cdt.SETTINGS.verbose``)
        ttest_threshold (float): threshold to stop the boostraps before
           ``nb_max_runs`` if the difference is significant
        nb_max_runs (int): Max number of bootstraps
        train_epochs (int): Number of epochs used for training
        test_epochs (int): Number of epochs used for evaluation

    .. note::
       Ref : Learning Functional Causal Models with Generative Neural Networks
       Olivier Goudet & Diviyan Kalainathan & Al.
       (https://arxiv.org/abs/1709.05321)

    """

    def __init__(self, nh=20, lr=0.01, nruns=6, njobs=None, gpus=None,
                 verbose=None, ttest_threshold=0.01,
                 nb_max_runs=16, train_epochs=1000, test_epochs=1000):
        """Init the model."""
        super(GNN, self).__init__()
        self.njobs = SETTINGS.get_default(njobs=njobs)
        self.gpus = SETTINGS.get_default(gpu=gpus)
        self.nh = nh
        self.lr = lr
        self.nruns = nruns
        self.nb_max_runs = nb_max_runs
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.ttest_threshold = ttest_threshold

    def predict_proba(self, dataset, idx=0):
        """Run multiple times GNN to estimate the causal direction.

        Args:
            dataset (torch.utils.data.Dataset or tuple): pair (x, y) to
               classify. Either a tuple or a torch dataset.

        Returns:
            float: Causal score of the pair (Value : 1 if a->b and -1 if b->a)
        """
        if isinstance(dataset, Dataset):
            data = dataset
        else:
            data = SimpleDataset(th.Tensor(scale(th.stack([th.Tensor(i).view(-1)
                                                           for i in dataset], 1))))
        ttest_criterion = TTestCriterion(
            max_iter=self.nb_max_runs, runs_per_iter=self.nruns,
            threshold=self.ttest_threshold)

        AB = []
        BA = []

        while ttest_criterion.loop(AB, BA):
            if self.njobs != 1:
                result_pair = parallel_run(GNN_instance, data, njobs=self.njobs,
                                           gpus=self.gpus, verbose=self.verbose,
                                           train_epochs=self.train_epochs,
                                           test_epochs=self.test_epochs,
                                           nruns=self.nruns)
            else:
                result_pair = [GNN_instance(data, device=SETTINGS.default_device,
                                            verbose=self.verbose,
                                            train_epochs=self.train_epochs,
                                            test_epochs=self.test_epochs)
                               for run in range(ttest_criterion.iter,
                                                ttest_criterion.iter +
                                                self.nruns)]
            AB.extend([runpair[0] for runpair in result_pair])
            BA.extend([runpair[1] for runpair in result_pair])

        if self.verbose:
            print("{} P-value after {} runs : {}".format(idx,
                                                         ttest_criterion.iter,
                                                         ttest_criterion.p_value))

        score_AB = np.mean(AB)
        score_BA = np.mean(BA)

        return (score_BA - score_AB) / (score_BA + score_AB)

    def orient_graph(self, df_data, graph, printout=None, **kwargs):
        """Orient an undirected graph using the pairwise method defined by the subclass.

        The pairwise method is ran on every undirected edge.

        Args:
            df_data (pandas.DataFrame or MetaDataset): Data (check cdt.utils.io.MetaDataset)
            graph (networkx.Graph): Graph to orient
            printout (str): (optional) Path to file where to save temporary results

        Returns:
            networkx.DiGraph: a directed graph, which might contain cycles

        .. note::
           This function is an override of the base class, in order to be able
           to use the torch.utils.data.Dataset classes

        .. warning::
           Requirement : Name of the nodes in the graph correspond to name of
           the variables in df_data
        """
        if isinstance(graph, nx.DiGraph):
            edges = [a for a in list(graph.edges()) if (a[1], a[0]) in list(graph.edges())]
            oriented_edges = [a for a in list(graph.edges()) if (a[1], a[0]) not in list(graph.edges())]
            for a in edges:
                if (a[1], a[0]) in list(graph.edges()):
                    edges.remove(a)
            output = nx.DiGraph()
            for i in oriented_edges:
                output.add_edge(*i)

        elif isinstance(graph, nx.Graph):
            edges = list(graph.edges())
            output = nx.DiGraph()

        else:
            raise TypeError("Data type not understood.")

        res = []
        for idx, (a, b) in enumerate(edges):
            if isinstance(df_data, DataFrame):
                dataset = SimpleDataset(th.Tensor(df_data[[a, b]].values))
                weight = self.predict_proba(dataset, idx=idx, **kwargs)
            elif isinstance(df_data, MetaDataset):
                weight = self.predict_proba(df_data.dataset(a, b),
                                            idx=idx, **kwargs)
            else:
                raise TypeError("Data type not understood.")
            if weight > 0:  # a causes b
                output.add_edge(a, b, weight=weight)
            elif weight < 0:
                output.add_edge(b, a, weight=abs(weight))
            if printout is not None:
                res.append([str(a) + '-' + str(b), weight])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)

        for node in list(df_data.columns.values):
            if node not in output.nodes():
                output.add_node(node)

        return output
