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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import scale
from .model import PairwiseModel
from ...utils.loss import MMDloss
from ...utils.Settings import SETTINGS
from ...utils.parallel import parallel_run
from ...utils.io import MetaDataset


class GNN_model(th.nn.Module):
    """Torch model for the GNN structure.

    Args:
        batch_size (int): size of the batch going to be fed to the model
        nh (int): Number of hidden units in the hidden layer
        lr (float): Learning rate of the Model
        train_epochs (int): Number of train epochs
        test_epochs (int): Number of test epochs
        idx (int): Index (for printing purposes)
        verbose (bool): Verbosity of the model
        dataloader_workers (int): Number of workers for dataset loading
        device (str): device on with the algorithm is going to be run on
    """

    def __init__(self, batch_size, nh=20, lr=0.01, train_epochs=1000,
                 test_epochs=1000, idx=0, verbose=None,
                 dataloader_workers=0, **kwargs):
        """Build the Torch graph.

        """
        super(GNN_model, self).__init__()
        self.l1 = th.nn.Linear(2, nh)
        self.l2 = th.nn.Linear(nh, 1)
        self.register_buffer('noise', th.Tensor(batch_size, 1))
        self.act = th.nn.ReLU()
        self.criterion = MMDloss(batch_size)
        self.layers = th.nn.Sequential(self.l1, self.act, self.l2)
        self.batch_size = batch_size
        self.lr = lr
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.idx = idx
        self.dataloader_workers = dataloader_workers

    def forward(self, x):
        """Pass data through the net structure.
        Args:
            x (torch.Tensor): input data: shape (:,1)

        Returns:
            torch.Tensor: Output of the shallow net
        """
        self.noise.normal_()
        return self.layers(th.cat([x, self.noise], 1))

    def run(self, dataset):
        """Run the GNN on a pair x,y of FloatTensor data.

        Args:
            dataset (torch.utils.data.Dataset): True data; First element is the cause

        Returns:
            torch.Tensor: Score of the configuration

        """
        optim = th.optim.Adam(self.parameters(), lr=self.lr)
        teloss = 0
        pbar = trange(self.train_epochs + self.test_epochs,
                      disable=not self.verbose)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, drop_last=True,
                                num_workers=self.dataloader_workers)
        for epoch in pbar:
            for i, (x, y) in enumerate(dataloader):
                optim.zero_grad()
                pred = self.forward(x)
                loss = self.criterion(pred, y)
                if epoch < self.train_epochs:
                    loss.backward()
                    optim.step()
                else:
                    teloss += loss.data

                # print statistics
                if not epoch % 50 and i == 0:
                    pbar.set_postfix(idx=self.idx, score=loss.item())

        return teloss.cpu().numpy() / self.test_epochs

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


def GNN_instance(data, batch_size=-1, idx=0, device=None, nh=20, **kwargs):
    """Run an instance of GNN, testing causal direction.

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param pair_idx: print purposes
    :param run: numner of the run (for GPU dispatch)
    :param device: device on with the algorithm is going to be run on.
    :return:
    """
    if batch_size == -1:
        batch_size = data.__len__()
    device = SETTINGS.get_default(device=device)
    GNNXY = GNN_model(batch_size, nh=nh, **kwargs).to(device)
    GNNYX = GNN_model(batch_size, nh=nh, **kwargs).to(device)
    GNNXY.reset_parameters()
    GNNYX.reset_parameters()
    if isinstance(data, Dataset):
        XY = GNNXY.run(data.to(device, flip=False))
        YX = GNNYX.run(data.to(device, flip=True))
    else:
        XY = GNNXY.run(TensorDataset(data[0].to(device), data[1].to(device)))
        YX = GNNYX.run(TensorDataset(data[1].to(device), data[0].to(device)))
    return [XY, YX]


class GNN(PairwiseModel):
    """Shallow Generative Neural networks.

    **Description:** Pairwise variant of the CGNN approach,
    Models the causal directions x->y and y->x with a 1-hidden layer neural
    network and a MMD loss. The causal direction is considered as the best-fit
    between the two causal directions.

    **Data Type:** Continuous

    **Assumptions:** The class of generative models is not restricted with a
    hard contraint, but with the hyperparameter ``nh``. This algorithm greatly
    benefits from bootstrapped runs (nruns >=12 recommended), and is very
    computationnally heavy. GPUs are recommended.

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
        batch_size (int): batch size, defaults to full-batch
        train_epochs (int): Number of epochs used for training
        test_epochs (int): Number of epochs used for evaluation
        dataloader_workers (int): how many subprocesses to use for data
           loading. 0 means that the data will be loaded in the main
           process. (default: 0)

    .. note::
       Ref : Learning Functional Causal Models with Generative Neural Networks
       Olivier Goudet & Diviyan Kalainathan & Al.
       (https://arxiv.org/abs/1709.05321)

    Example:
        >>> from cdt.causality.pairwise import GNN
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from cdt.data import load_dataset
        >>> data, labels = load_dataset('tuebingen')
        >>> obj = GNN()
        >>>
        >>> # This example uses the predict() method
        >>> output = obj.predict(data)
        >>>
        >>> # This example uses the orient_graph() method. The dataset used
        >>> # can be loaded using the cdt.data module
        >>> data, graph = load_dataset("sachs")
        >>> output = obj.orient_graph(data, nx.Graph(graph))
        >>>
        >>> #To view the directed graph run the following command
        >>> nx.draw_networkx(output, font_size=8)
        >>> plt.show()
    """

    def __init__(self, nh=20, lr=0.01, nruns=6, njobs=None, gpus=None,
                 verbose=None, batch_size=-1,
                 train_epochs=1000, test_epochs=1000,
                 dataloader_workers=0):
        """Init the model."""
        super(GNN, self).__init__()
        self.njobs = SETTINGS.get_default(njobs=njobs)
        self.gpus = SETTINGS.get_default(gpu=gpus)
        self.nh = nh
        self.lr = lr
        self.nruns = nruns
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.dataloader_workers = dataloader_workers

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
            data = [th.Tensor(scale(th.Tensor(i).view(-1, 1)))
                       for i in dataset]

        AB = []
        BA = []

        if self.gpus > 1:
            result_pair = parallel_run(GNN_instance, data, njobs=self.njobs,
                                       gpus=self.gpus, verbose=self.verbose,
                                       train_epochs=self.train_epochs,
                                       test_epochs=self.test_epochs,
                                       nruns=self.nruns,
                                       batch_size=self.batch_size,
                                       dataloader_workers=self.dataloader_workers)
        else:
            result_pair = [GNN_instance(data, device=SETTINGS.default_device,
                                        verbose=self.verbose,
                                        train_epochs=self.train_epochs,
                                        test_epochs=self.test_epochs,
                                        batch_size=self.batch_size,
                                        dataloader_workers=self.dataloader_workers)
                           for run in range(self.nruns)]
        AB.extend([runpair[0] for runpair in result_pair])
        BA.extend([runpair[1] for runpair in result_pair])

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
        if isinstance(df_data, DataFrame):
            var_names = list(df_data.columns)
        elif isinstance(df_data, MetaDataset):
            var_names = df_data.get_names()

        for idx, (a, b) in enumerate(edges):
            if isinstance(df_data, DataFrame):
                dataset = (th.Tensor(scale(df_data[a].values)).view(-1, 1),
                           th.Tensor(scale(df_data[b].values)).view(-1, 1))
                weight = self.predict_proba(dataset, idx=idx, **kwargs)
            elif isinstance(df_data, MetaDataset):
                weight = self.predict_proba(df_data.dataset(a, b, scale=True),
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

        for node in var_names:
            if node not in output.nodes():
                output.add_node(node)

        return output
