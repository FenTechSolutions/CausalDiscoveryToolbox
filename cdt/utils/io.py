"""Formatting and import functions.

Author: Diviyan Kalainathan
Date : 2/06/17

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
from pandas import DataFrame, read_csv
from numpy import array
from sklearn.preprocessing import scale as scaler
import networkx as nx
from torch.utils.data import Dataset
import torch as th
from collections import OrderedDict
from copy import deepcopy


def read_causal_pairs(filename, scale=False, **kwargs):
    """Convert a ChaLearn Cause effect pairs challenge format into numpy.ndarray.

    Args:
        filename (str or pandas.DataFrame): path of the file to read or DataFrame containing the data
        scale (bool): Scale the data
        \**kwargs: parameters to be passed to pandas.read_csv

    Returns:
        pandas.DataFrame: Dataframe composed of (SampleID, a (numpy.ndarray) , b (numpy.ndarray))

    Examples:
        >>> from cdt.utils import read_causal_pairs
        >>> data = read_causal_pairs('file.tsv', scale=True, sep='\\t')
    """
    def convert_row(row, scale):
        """Convert a CCEPC row into numpy.ndarrays.

        :param row:
        :type row: pandas.Series
        :return: tuple of sample ID and the converted data into numpy.ndarrays
        :rtype: tuple
        """
        a = row["A"].split(" ")
        b = row["B"].split(" ")

        if a[0] == "":
            a.pop(0)
            b.pop(0)
        if a[-1] == "":
            a.pop(-1)
            b.pop(-1)

        a = array([float(i) for i in a])
        b = array([float(i) for i in b])
        if scale:
            a = scaler(a)
            b = scaler(b)
        return row['SampleID'], a, b
    if isinstance(filename, str):
        data = read_csv(filename, **kwargs)
    elif isinstance(filename, DataFrame):
        data = filename
    else:
        raise TypeError("Type not supported.")
    conv_data = []

    for idx, row in data.iterrows():
        conv_data.append(convert_row(row, scale))
    df = DataFrame(conv_data, columns=['SampleID', 'A', 'B'])
    df = df.set_index("SampleID")
    return df


def read_adjacency_matrix(filename, directed=True, **kwargs):
    """Read a file (containing an adjacency matrix) and convert it into a
    directed or undirected networkx graph.

    :param filename: file to read or DataFrame containing the data
    :type filename: str or pandas.DataFrame
    :param directed: Return directed graph
    :type directed: bool
    :param kwargs: extra parameters to be passed to pandas.read_csv
    :return: networkx graph containing the graph.
    :rtype: **networkx.DiGraph** or **networkx.Graph** depending on the
      ``directed`` parameter.

    Examples:
        >>> from cdt.utils import read_adjacency_matrix
        >>> data = read_causal_pairs('graph_file.csv', directed=False)
    """
    if isinstance(filename, str):
        data = read_csv(filename, **kwargs)
    elif isinstance(filename, DataFrame):
        data = filename
    else:
        raise TypeError("Type not supported.")
    if directed:
        return nx.relabel_nodes(nx.DiGraph(data.values),
                                {idx: i for idx, i in enumerate(data.columns)})
    else:
        return nx.relabel_nodes(nx.Graph(data.values),
                                {idx: i for idx, i in enumerate(data.columns)})


def read_list_edges(filename, directed=True, **kwargs):
    """Read a file (containing list of edges) and convert it into a directed
    or undirected networkx graph.

    :param filename: file to read or DataFrame containing the data
    :type filename: str or pandas.DataFrame
    :param directed: Return directed graph
    :type directed: bool
    :param kwargs: extra parameters to be passed to pandas.read_csv
    :return: networkx graph containing the graph.
    :rtype: **networkx.DiGraph** or **networkx.Graph** depending on the
      ``directed`` parameter.

    Examples:
        >>> from cdt.utils import read_adjacency_matrix
        >>> data = read_causal_pairs('graph_file.csv', directed=False)
    """
    if isinstance(filename, str):
        data = read_csv(filename, **kwargs)
    elif isinstance(filename, DataFrame):
        data = filename
    else:
        raise TypeError("Type not supported.")
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    if len(data.columns) == 3:
        data.columns = ['Cause', 'Effect', 'Score']
    else:
        data.columns = ['Cause', 'Effect']

    for idx, row in data.iterrows():
        try:
            score = row["Score"]
        except KeyError:
            score = 1
        graph.add_edge(row['Cause'], row["Effect"], weight=score)

    return graph


# class SimpleDataset(Dataset):
#     def __init__(self, data, device=None):
#         super(SimpleDataset, self).__init__()
#         self.data = data
#         if device is not None:
#             self.data = data.to(device)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def to(self, device):
#         return SimpleDataset(self.data, device)


class PairwiseDataset(Dataset):
    """Dataset class for pairwise methods.

    Class can be overriden to have more specific dataloaders,
    in case of large amounts of data.

    Args:
        a (array-like): Variable 1
        b (array-like): Variable 2
        device (str): device on which the data has to be sent.
           Data must be of type `torch.Tensor` if `device` is specified.
        flip (bool): return the data in the reversed order.

    """
    def __init__(self, a, b, device=None, flip=False):
        super(PairwiseDataset, self).__init__()
        self.a = a
        self.b = b
        if device is not None:
            self.a = a.to(device)
            self.b = b.to(device)
        self.flip = flip

    def __len__(self):
        return len(self.a)

    def __getitem__(self, index):
        if self.flip:
            return self.b[index], self.a[index]
        else:
            return self.a[index], self.b[index]

    def to(self, device, flip=False):
        """ Produce a copy of the dataset on a device

        Args:
            device (str): device on which the data has to be sent.
               Data must be of type `torch.Tensor` if `device` is specified.
            flip (bool): return the data in the reversed order.

        Returns:
            cdt.utils.io.PairwiseDataset: the new dataset on device
        """
        return PairwiseDataset(self.a, self.b, device, flip)


class MetaDataset(Dataset):
    """Meta-Dataset class for `torch.utils.data.DataLoader`.

    Class can be overriden to have more specific dataloaders,
    in case of large amounts of data.

    Args:
        data (pandas.DataFrame or array-like): input data.
        names (dict): dict of `variable_name:column_index` of the data. If not
           specified, data has to be a pandas.DataFrame.
        device (str): device on which the data has to be sent.
           Data must be of type `torch.Tensor` if `device` is specified.
        scale (bool): scale the data with 0 mean and 1 variance.

    """
    def __init__(self, data, names=None, device=None, scale=True):
        super(MetaDataset, self).__init__()
        if names is not None:
            self.names = names
        else:
            try:
                assert isinstance(data, DataFrame)
            except AssertionError:
                raise TypeError('If names is not specified, \
                data has to be a pandas.DataFrame')
            self.names = OrderedDict([(i, idx) for idx,
                                      i in enumerate(data.columns)])

        if isinstance(data, DataFrame):
            data = data.values

        if scale:
            self.data = th.Tensor(scaler(data))
        else:
            self.data = th.Tensor(data)

        if device is not None:
            self.data = self.data.to(device)

    def get_names(self):
        """Get the column names in the corresponding order"""
        return list(self.names.keys())

    def to(self, device):
        """Produce a copy of the dataset on a device

        Args:
            device (str): device on which the data has to be sent.

        Returns:
            cdt.utils.io.MetaDataset: the new dataset on device
        """
        cpy = deepcopy(self)
        cpy.data = cpy.data.to(device)
        return cpy  # MetaDataset(self.data, self.names, device)

    def __len__(self):
        return len(self.data)

    def __featurelen__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def dataset(self, a, b, scale=False, shape=(-1, 1)):
        """Produce a PairwiseDataset of two variables out of the data.

        Args:
            a (str): Name of the first variable
            b (str): Name of the second variable
            scale (bool): scale the data with 0 mean and 1 variance.
            shape (tuple): desired shape of `torch.Tensor` of `a` and `b`

        Returns:
            cdt.utils.io.MetaDataset: the new pairwise dataset
        """
        a = self.data[:, self.names[a]]
        b = self.data[:, self.names[b]]
        if scale:
            a = scaler(a)
            b = scaler(b)
        return PairwiseDataset(th.Tensor(a).view(*shape),
                               th.Tensor(b).view(*shape))
