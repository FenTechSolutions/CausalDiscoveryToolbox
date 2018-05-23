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

    :param filename: Name fo the file read
    :type filename: str
    :param scale: Scale the data
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

    data = read_csv(filename, **kwargs)
    conv_data = []

    for idx, row in data.iterrows():
        conv_data.append(convert_row(row, scale))
    df = DataFrame(conv_data, columns=['SampleID', 'A', 'B'])
    df = df.set_index("SampleID")
    return df


def read_adjacency_matrix(filename, directed=True, **kwargs):
    """Read a file (containing an adjacency matrix) and convert it into a directed or undirected networkx graph.

    :param filename: file to read
    :param directed: Return directed graph
    :param kwargs: extra parameters to be passed to pandas.read_csv
    """
    data = read_csv(filename, **kwargs)
    if directed:
        return nx.relabel_nodes(nx.DiGraph(data.values),
                                {idx: i for idx, i in enumerate(data.columns)})
    else:
        return nx.relabel_nodes(nx.Graph(data.values),
                                {idx: i for idx, i in enumerate(data.columns)})


def read_list_edges(filename, directed=True, **kwargs):
    """Read a file (containing list of edges) and convert it into a directed or undirected networkx graph.

    :param filename: file to be read, per default columns=Cause,Effect
    :param directed:
    :param kwargs:
    """
    data = read_csv(filename, **kwargs)
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
