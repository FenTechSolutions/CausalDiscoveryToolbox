"""Formatting and import functions.

Author: Diviyan Kalainathan
Date : 2/06/17

"""
from pandas import DataFrame, read_csv
from numpy import array
from sklearn.preprocessing import scale as scaler
import networkx as nx


def read_causal_pairs(filename, scale=True, **kwargs):
    """Convert a ChaLearn Cause effect pairs challenge format into numpy.ndarray.

    :param filename: path of the file to read or DataFrame containing the data
    :type filename: str or pandas.DataFrame
    :param scale: Scale the data
    :type scale: bool
    :param kwargs: parameters to be passed to pandas.read_csv
    :return: Dataframe composed of (SampleID, a (numpy.ndarray) , b (numpy.ndarray))
    :rtype: pandas.DataFrame
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

    for idx, row in data.iterrows():
        try:
            score = row["Score"]
        except KeyError:
            score = 1
        graph.add_edge(row['Cause'], row["Effect"], weight=score)

    return graph
