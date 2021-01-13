"""Structural Agnostic Model.

Author: Diviyan Kalainathan, Olivier Goudet
Date: 09/3/2018

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
import os
import numpy as np
import torch as th
import pandas as pd
import networkx as nx
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import scale
from .model import GraphModel
from ...utils.parallel import parallel_run
from ...utils.loss import notears_constr
from ...utils.torch import (ChannelBatchNorm1d, MatrixSampler,
                            Linear3D, ParallelBatchNorm1d,
                            SimpleMatrixConnection)
from ...utils.Settings import SETTINGS


class SAM_generators(th.nn.Module):
    """Ensemble of all the generators."""

    def permutation_matrix(self, skeleton, data_shape, max_dim):
        reshape_skeleton = th.zeros(self.nb_vars, int(data_shape[1]), max_dim)

        for channel in range(self.nb_vars):
            perm_matrix = skeleton[:, channel] * th.eye(data_shape[1],data_shape[1])
            skeleton_list = [i for i in th.unbind(perm_matrix, 1) if th.count_nonzero(i) > 0]
            perm_matrix = th.stack(skeleton_list, 1) if len(skeleton_list)>0 else th.zeros(data_shape[1], 1)
            reshape_skeleton[channel, :, :perm_matrix.shape[1]] = perm_matrix

        return reshape_skeleton

    def __init__(self, data_shape, nh, skeleton=None, cat_sizes=None, linear=False, numberHiddenLayersG=1):
        """Init the model."""
        super(SAM_generators, self).__init__()
        layers = []
        # Building skeleton
        self.sizes = cat_sizes
        self.linear = linear

        if cat_sizes is not None:
            nb_vars = len(cat_sizes)
            output_dim = max(cat_sizes)
            cat_reshape = th.zeros(nb_vars, sum(cat_sizes))
            for var, (cat, cumul) in enumerate(zip(cat_sizes, np.cumsum(cat_sizes))):
                cat_reshape[var, cumul-cat:cumul].fill_(1)
        else:
            nb_vars = data_shape[1]
            output_dim = 1
            cat_reshape = th.eye(nb_vars, nb_vars)

        self.nb_vars = nb_vars
        if skeleton is None:
            skeleton = 1 - th.eye(nb_vars, nb_vars)

        # Redimensioning the skeleton according to the categorical vars
        skeleton = cat_reshape.t() @ skeleton @ cat_reshape
        # torch 0.4.1
        max_dim = th.as_tensor(skeleton.sum(dim=0).max(), dtype=th.int)
        # torch 0.4.0
        # max_dim = skeleton.sum(dim=0).max()

        # Building the custom matrix for reshaping.
        reshape_skeleton = self.permutation_matrix(skeleton, data_shape, max_dim)

        if linear:
            self.input_layer = Linear3D(nb_vars, max_dim, output_dim,
                                        noise=True, batch_size=data_shape[0])
        else:
            self.input_layer = Linear3D(nb_vars, max_dim, nh, noise=True, batch_size=data_shape[0])
            layers.append(ChannelBatchNorm1d(nb_vars, nh))
            layers.append(th.nn.Tanh())


            for i in range(numberHiddenLayersG-1):
                layers.append(Linear3D(nb_vars, nh, nh))
                layers.append(ChannelBatchNorm1d(nb_vars, nh))
                layers.append(th.nn.Tanh())

            self.output_layer = Linear3D(nb_vars, nh, output_dim)
            # self.weights = Linear3D(data_shape[1], data_shape[1], 1)
            self.layers = th.nn.Sequential(*layers)

        self.register_buffer('skeleton', reshape_skeleton)
        self.register_buffer("categorical_matrix", cat_reshape)

    def forward(self, data, adj_matrix, drawn_neurons=None):
        """Forward through all the generators."""

        if self.linear:
            output = self.input_layer(data, self.categorical_matrix.t() @ adj_matrix, self.skeleton)
        else:
            output = self.output_layer(self.layers(self.input_layer(data,
                                                                self.categorical_matrix.t() @ adj_matrix,
                                                                self.skeleton)),
                                                                drawn_neurons)

        if self.sizes is None:
            return output.squeeze(2)
        else:
            return th.cat([th.nn.functional.softmax(output[:, idx, :i], dim=1)
                           if i>1 else output[:, idx, :i] for idx, i in enumerate(self.sizes)], 1)

    def reset_parameters(self):
        if not self.linear:
            self.output_layer.reset_parameters()
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.input_layer.reset_parameters()

    def apply_filter(self, skeleton, data_shape, device):

        skeleton = self.categorical_matrix.cpu().t() @ skeleton @ self.categorical_matrix.cpu()
        max_dim = skeleton.sum(dim=0).max()
        reshape_skeleton = self.permutation_matrix(skeleton,
                                                   data_shape,
                                                   max_dim).to(device)

        self.register_buffer('skeleton', reshape_skeleton)
        self.input_layer.apply_filter(th.cat([self.skeleton,
                                              th.ones(self.skeleton.shape[0],
                                                      self.skeleton.shape[1],
                                                      1).to(device)],2) )


class SAM_discriminator(th.nn.Module):
    """SAM discriminator."""

    def __init__(self, nfeatures, dnh, numberHiddenLayersD=2, mask=None):
        super(SAM_discriminator, self).__init__()
        self.nfeatures = nfeatures
        layers = []
        layers.append(th.nn.Linear(nfeatures, dnh))
        layers.append(ParallelBatchNorm1d(dnh))
        layers.append(th.nn.LeakyReLU(.2))
        for i in range(numberHiddenLayersD-1):
            layers.append(th.nn.Linear(dnh, dnh))
            layers.append(ParallelBatchNorm1d(dnh))
            layers.append(th.nn.LeakyReLU(.2))

        layers.append(th.nn.Linear(dnh, 1))
        self.layers = th.nn.Sequential(*layers)

        if mask is None:
            mask = th.eye(nfeatures, nfeatures)
        self.register_buffer("mask", mask.unsqueeze(0))

    def forward(self, input, obs_data=None):
        if obs_data is not None:
            return self.layers(obs_data.unsqueeze(1) * (1 - self.mask) + input.unsqueeze(1) * self.mask)
        else:
            return self.layers(input)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()



def run_SAM(in_data, skeleton=None, is_mixed=False, device="cpu",
            train=10000, test=1,
            batch_size=-1, lr_gen=.001,
            lr_disc=.01, lambda1=0.001, lambda2=0.0000001, nh=None, dnh=None,
            verbose=True, losstype="fgan", functionalComplexity="n_hidden_units",
            sampletype="sigmoidproba",
            dagstart=0, dagloss=False,
            dagpenalization=0.05, dagpenalization_increase=0.0,
            categorical_threshold=50,
            linear=False, numberHiddenLayersG=2, numberHiddenLayersD=2, idx=0):

    list_nodes = list(in_data.columns)
    if is_mixed:
        onehotdata = []
        for i in range(len(list_nodes)):
            # print(pd.get_dummies(in_data.iloc[:, i]).values.shape[1])
            if pd.get_dummies(in_data.iloc[:, i]).values.shape[1] < categorical_threshold:
                onehotdata.append(pd.get_dummies(in_data.iloc[:, i]).values)
            else:
                onehotdata.append(scale(in_data.iloc[:, [i]].values))
        cat_sizes = [i.shape[1] for i in onehotdata]

        data = np.concatenate(onehotdata, 1)
    else:
        data = scale(in_data[list_nodes].values)
        cat_sizes = None

    nb_var = len(list_nodes)
    data = data.astype('float32')
    data = th.from_numpy(data).to(device)
    if batch_size == -1:
        batch_size = data.shape[0]

    lambda1 = lambda1/data.shape[0]
    lambda2 = lambda2/data.shape[0]


    rows, cols = data.size()
    # Get the list of indexes to ignore
    if skeleton is not None:
        skeleton = th.from_numpy(skeleton.astype('float32'))

    sam = SAM_generators((batch_size, cols), nh, skeleton=skeleton,
                         cat_sizes=cat_sizes, linear=linear, numberHiddenLayersG=numberHiddenLayersG).to(device)

    sam.reset_parameters()
    g_optimizer = th.optim.Adam(list(sam.parameters()), lr=lr_gen)

    if losstype != "mse":
        discriminator = SAM_discriminator(cols, dnh, numberHiddenLayersD,
                                          mask=sam.categorical_matrix,).to(device)
        discriminator.reset_parameters()
        d_optimizer = th.optim.Adam(discriminator.parameters(), lr=lr_disc)
        criterion = th.nn.BCEWithLogitsLoss()
    else:
        criterion = th.nn.MSELoss()
        disc_loss = th.zeros(1)


    if sampletype == "sigmoid":
        graph_sampler = SimpleMatrixConnection(len(list_nodes), mask=skeleton).to(device)
    elif sampletype == "sigmoidproba":
        graph_sampler = MatrixSampler(len(list_nodes), mask=skeleton, gumble=False).to(device)
    elif sampletype == "gumbleproba":
        graph_sampler = MatrixSampler(len(list_nodes), mask=skeleton, gumble=True).to(device)
    else:
        raise ValueError('Unknown Graph sampler')

    graph_sampler.weights.data.fill_(2)

    graph_optimizer = th.optim.Adam(graph_sampler.parameters(), lr=lr_gen)

    if not linear and functionalComplexity=="n_hidden_units":
        neuron_sampler = MatrixSampler((nh, len(list_nodes)), mask=False, gumble=True).to(device)
        neuron_optimizer = th.optim.Adam(list(neuron_sampler.parameters()),lr=lr_gen)

    _true = th.ones(1).to(device)
    _false = th.zeros(1).to(device)
    output = th.zeros(len(list_nodes), len(list_nodes)).to(device)

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)


    # RUN
    if verbose:
        pbar = tqdm(range(train + test))
    else:
        pbar = range(train+test)
    for epoch in pbar:
        for i_batch, batch in enumerate(data_iterator):

            if losstype != "mse":
                d_optimizer.zero_grad()

            # Train the discriminator

            drawn_graph = graph_sampler()

            if not linear and functionalComplexity=="n_hidden_units":
                drawn_neurons = neuron_sampler()


            if linear or functionalComplexity!="n_hidden_units":
                generated_variables = sam(batch, drawn_graph)
            else:
                generated_variables = sam(batch, drawn_graph, drawn_neurons)

            if losstype != "mse":
                disc_vars_d = discriminator(generated_variables.detach(), batch)
                true_vars_disc = discriminator(batch)

                if losstype == "gan":
                    disc_loss = sum([criterion(gen, _false.expand_as(gen)) for gen in disc_vars_d]) / nb_var \
                                     + criterion(true_vars_disc, _true.expand_as(true_vars_disc))
                    # Gen Losses per generator: multiply py the number of channels
                elif losstype == "fgan":

                    disc_loss = th.mean(th.exp(disc_vars_d - 1), [0, 2]).sum() / nb_var - th.mean(true_vars_disc)

                disc_loss.backward()
                d_optimizer.step()


            ### OPTIMIZING THE GENERATORS
            g_optimizer.zero_grad()
            graph_optimizer.zero_grad()

            if not linear and functionalComplexity=="n_hidden_units":
                neuron_optimizer.zero_grad()

            if losstype == "mse":
                gen_loss = criterion(generated_variables, batch)
            else:
                disc_vars_g = discriminator(generated_variables, batch)

                if losstype == "gan":
                    # Gen Losses per generator: multiply py the number of channels
                    gen_loss = sum([criterion(gen,
                                              _true.expand_as(gen))
                                    for gen in disc_vars_g])
                elif losstype == "fgan":
                    gen_loss = -th.mean(th.exp(disc_vars_g - 1), [0, 2]).sum()

            filters = graph_sampler.get_proba()
            struc_loss = lambda1*drawn_graph.sum()

            if linear :
                func_loss = 0
            else :
                if functionalComplexity=="n_hidden_units":
                    func_loss = lambda2*drawn_neurons.sum()


                elif functionalComplexity=="l2_norm":
                    l2_reg = th.Tensor([0.]).to(device)
                    for param in sam.parameters():
                        l2_reg += th.norm(param)

                    func_loss = lambda2*l2_reg

            regul_loss = struc_loss + func_loss


            # Optional: prune edges and sam parameters before dag search

            if dagloss and epoch > train * dagstart:
                dag_constraint = notears_constr(filters*filters)
                #dag_constraint = notears_constr(drawn_graph)

                loss = gen_loss + regul_loss + (dagpenalization + (epoch - train * dagstart) * dagpenalization_increase) * dag_constraint
            else:
                loss = gen_loss + regul_loss
            if verbose and epoch % 20 == 0 and i_batch == 0:
                pbar.set_postfix(gen=gen_loss.item()/cols,
                                 disc=disc_loss.item(),
                                 regul_loss=regul_loss.item(),
                                 tot=loss.item())

            if epoch < train + test - 1:
                loss.backward()

            if epoch >= train:
                output.add_(filters.data)

            g_optimizer.step()
            graph_optimizer.step()
            if not linear and functionalComplexity=="n_hidden_units":
                neuron_optimizer.step()

    return output.div_(test).cpu().numpy()
    # Evaluate total effect with final DAG


class SAM(GraphModel):
    """SAM Algorithm.

    **Description:** Structural Agnostic Model is an causal discovery algorithm
    for DAG recovery leveraging both distributional asymetries and conditional
    independencies. the first version of SAM without DAG constraint is available
    as ``SAMv1``.

    **Data Type:** Continuous, (Mixed - Experimental)

    **Assumptions:** The class of generative models is not restricted with a
    hard contraint, but with soft constraints parametrized with the ``lambda1``
    and ``lambda2`` parameters, with gumbel softmax sampling. This algorithms greatly
    benefits from bootstrapped runs (nruns >=8 recommended).
    GPUs are recommended but not compulsory. The output is a DAG, but may need a
    thresholding as the output is averaged over multiple runs.

    Args:
        lr (float): Learning rate of the generators
        dlr (float): Learning rate of the discriminator
        mixed_data (bool): Experimental -- Enable for mixed-type datasets
        lambda1 (float): L0 penalization coefficient on the causal filters
        lambda2 (float): L2 penalization coefficient on the weights of the
           neural network
        nh (int): Number of hidden units in the generators' hidden layers
           (regularized with lambda2)
        dnh (int): Number of hidden units in the discriminator's hidden layers
        train_epochs (int): Number of training epochs
        test_epochs (int): Number of test epochs (saving and averaging
           the causal filters)
        batch_size (int): Size of the batches to be fed to the SAM model
           Defaults to full-batch
        losstype (str): type of the loss to be used (either 'fgan' (default),
           'gan' or 'mse')
        dagloss (bool): Activate the DAG with No-TEARS constraint
        dagstart (float): Controls when the DAG constraint is to be introduced
           in the training (float ranging from 0 to 1, 0 denotes the start of
           the training and 1 the end)
        dagpenalisation (float): Initial value of the DAG constraint
        dagpenalisation_increase (float): Increase incrementally at each epoch
           the coefficient of the constraint
        functional_complexity (str): Type of functional complexity penalization
           (choose between 'l2_norm' and 'n_hidden_units')
        hlayers (int): Defines the number of hidden layers in the generators
        dhlayers (int): Defines the number of hidden layers in the discriminator
        sampling_type (str): Type of sampling used in the structural gates of the
           model (choose between 'sigmoid', 'sigmoid_proba' and 'gumble_proba')
        linear (bool): If true, all generators are set to be linear generators
        nruns (int): Number of runs to be made for causal estimation
               Recommended: >=32 for optimal performance
        njobs (int): Numbers of jobs to be run in Parallel
               Recommended: 1 if no GPU available, 2*number of GPUs else
        gpus (int): Number of available GPUs for the algorithm
        verbose (bool): verbose mode

    .. note::
       Ref: Kalainathan, Diviyan & Goudet, Olivier & Guyon, Isabelle &
       Lopez-Paz, David & Sebag, Michèle. (2018). Structural Agnostic Modeling:
       Adversarial Learning of Causal Graphs.

    Example:
        >>> import networkx as nx
        >>> from cdt.causality.graph import SAM
        >>> from cdt.data import load_dataset
        >>> data, graph = load_dataset("sachs")
        >>> obj = SAM()
        >>> #The predict() method works without a graph, or with a
        >>> #directed or undirected graph provided as an input
        >>> output = obj.predict(data)    #No graph provided as an argument
        >>>
        >>> output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
        >>>
        >>> output = obj.predict(data, graph)  #With a directed graph
        >>>
        >>> #To view the graph created, run the below commands:
        >>> nx.draw_networkx(output, font_size=8)
        >>> plt.show()
    """

    def __init__(self, lr=0.01, dlr=0.001, mixed_data=False,
                 lambda1=10, lambda2=0.001,
                 nh=20, dnh=200,
                 train_epochs=3000, test_epochs=1000, batch_size=-1,
                 losstype="fgan", dagloss=True, dagstart=0.5,
                 dagpenalization=0,
                 dagpenalization_increase=0.01,
                 functional_complexity='l2_norm', hlayers=2, dhlayers=2,
                 sampling_type='sigmoidproba', linear=False, nruns=8,
                 njobs=None, gpus=None, verbose=None):

        """Init and parametrize the SAM model."""
        super(SAM, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.mixed_data = mixed_data
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batch_size = batch_size
        self.dagstart = dagstart
        self.dagloss = dagloss
        self.dagpenalization = dagpenalization
        self.dagpenalization_increase = dagpenalization_increase
        self.losstype = losstype
        self.functionalComplexity = functional_complexity
        self.sampletype = sampling_type
        self.linear = linear
        self.numberHiddenLayersG = hlayers
        self.numberHiddenLayersD = dhlayers
        self.njobs = SETTINGS.get_default(njobs=njobs)
        self.gpus = SETTINGS.get_default(gpu=gpus)
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.nruns = nruns

    def predict(self, data, graph=None,
                return_list_results=False):
        """Execute SAM on a dataset given a skeleton or not.

        Args:
            data (pandas.DataFrame): Observational data for estimation of causal relationships by SAM
            skeleton (numpy.ndarray): A priori knowledge about the causal relationships as an adjacency matrix.
                      Can be fed either directed or undirected links.
        Returns:
            networkx.DiGraph: Graph estimated by SAM, where A[i,j] is the term
            of the ith variable for the jth generator.
        """
        if graph is not None:
            skeleton = th.Tensor(nx.adjacency_matrix(graph,
                                                     nodelist=list(data.columns)).todense())
        else:
            skeleton = None

        assert self.nruns > 0
        if self.gpus == 0:
            results = [run_SAM(data, skeleton=skeleton,
                               lr_gen=self.lr, lr_disc=self.dlr,
                               is_mixed=self.mixed_data,
                               lambda1=self.lambda1, lambda2=self.lambda2,
                               nh=self.nh, dnh=self.dnh,
                               train=self.train,
                               test=self.test, batch_size=self.batch_size,
                               dagstart=self.dagstart,
                               dagloss=self.dagloss,
                               dagpenalization=self.dagpenalization,
                               dagpenalization_increase=self.dagpenalization_increase,
                               losstype=self.losstype,
                               functionalComplexity=self.functionalComplexity,
                               sampletype=self.sampletype,
                               linear=self.linear,
                               numberHiddenLayersD=self.numberHiddenLayersD,
                               numberHiddenLayersG=self.numberHiddenLayersG,
                               device='cpu') for i in range(self.nruns)]
        else:
            results = parallel_run(run_SAM, data, skeleton=skeleton,
                                   nruns=self.nruns,
                                   njobs=self.njobs, gpus=self.gpus,
                                   lr_gen=self.lr, lr_disc=self.dlr,
                                   is_mixed=self.mixed_data,
                                   lambda1=self.lambda1, lambda2=self.lambda2,
                                   nh=self.nh, dnh=self.dnh,
                                   train=self.train,
                                   test=self.test, batch_size=self.batch_size,
                                   dagstart=self.dagstart,
                                   dagloss=self.dagloss,
                                   dagpenalization=self.dagpenalization,
                                   dagpenalization_increase=self.dagpenalization_increase,
                                   losstype=self.losstype,
                                   functionalComplexity=self.functionalComplexity,
                                   sampletype=self.sampletype,
                                   linear=self.linear,
                                   numberHiddenLayersD=self.numberHiddenLayersD,
                                   numberHiddenLayersG=self.numberHiddenLayersG)
        list_out = [i for i in results if not np.isnan(i).any()]
        try:
            assert len(list_out) > 0
        except AssertionError as e:
            print("All solutions contain NaNs")
            raise(e)
        W = sum(list_out)/len(list_out)
        return nx.relabel_nodes(nx.DiGraph(W),
                                {idx: i for idx,
                                 i in enumerate(data.columns)})

    def orient_directed_graph(self, *args, **kwargs):
        """Orient a (partially directed) graph."""
        return self.predict(*args, **kwargs)

    def orient_undirected_graph(self, *args, **kwargs):
        """Orient a undirected graph."""
        return self.predict(*args, **kwargs)

    def create_graph_from_data(self, *args, **kwargs):
        """Estimate a causal graph out of observational data."""
        return self.predict(*args, **kwargs)
